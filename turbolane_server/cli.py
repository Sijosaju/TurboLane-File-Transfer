#!/usr/bin/env python3
"""
turbolane-server — CLI entry point

Subcommands
───────────
  start   Start the TurboLane file transfer server (receiver)
  send    Send a file to a running TurboLane server
  status  Query a running server for its current transfer status

Usage examples
──────────────
  # Start receiver server on port 9000, save files to ./received/
  turbolane-server start --port 9000 --output-dir ./received

  # Send a file using 6 initial streams, allow TurboLane to scale 1-32
  turbolane-server send /data/large_dataset.tar \\
      --host 192.168.1.50 --port 9000 \\
      --streams 6 --min-streams 1 --max-streams 32 \\
      --model-dir models/dci

  # Query status of a running server
  turbolane-server status --host 192.168.1.50 --port 9000
"""

import argparse
import logging
import sys
import os
import socket
import json

# ---------------------------------------------------------------------------
# Logging setup (called before anything else so imports log correctly)
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt   = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)
    # Quiet noisy loggers unless verbose
    if not verbose:
        logging.getLogger("turbolane.rl.agent").setLevel(logging.WARNING)
        logging.getLogger("turbolane.policies.federated").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Subcommand: start
# ---------------------------------------------------------------------------

def cmd_start(args: argparse.Namespace) -> int:
    """Start the TurboLane receiver server."""
    from turbolane_server.server import TurboLaneServer

    server = TurboLaneServer(
        host=args.host,
        port=args.port,
        output_dir=args.output_dir,
    )
    server.start()   # blocks
    return 0


# ---------------------------------------------------------------------------
# Subcommand: send
# ---------------------------------------------------------------------------

def cmd_send(args: argparse.Namespace) -> int:
    """Send a file to a TurboLane receiver."""
    from turbolane_server.sender import TurboLaneSender

    try:
        sender = TurboLaneSender(
            file_path=args.file,
            receiver_host=args.host,
            receiver_port=args.port,
            initial_streams=args.streams,
            min_streams=args.min_streams,
            max_streams=args.max_streams,
            model_dir=args.model_dir,
            monitor_interval=args.interval,
            timeout=args.timeout,
        )
    except FileNotFoundError as exc:
        print(f"\n  Error: {exc}", file=sys.stderr)
        return 1

    success = sender.run()
    return 0 if success else 1


# ---------------------------------------------------------------------------
# Subcommand: status
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> int:
    """
    Connect to a running TurboLane server and print its status.

    Sends a STATUS_REQ control frame over a fresh TCP connection.
    The server responds with STATUS_RESP containing a JSON payload.
    """
    from turbolane_server.protocol import (
        MessageType,
        encode_frame,
        decode_meta,
        recv_frame,
    )

    print(f"\n  Querying {args.host}:{args.port} ...\n")
    try:
        with socket.create_connection((args.host, args.port), timeout=5.0) as sock:
            req = encode_frame(msg_type=MessageType.STATUS_REQ)
            sock.sendall(req)
            hdr, payload = recv_frame(sock)

            if hdr["msg_type"] == MessageType.STATUS_RESP and payload:
                data = decode_meta(payload)
                _print_status(data)
            elif hdr["msg_type"] == MessageType.BUSY:
                print("  Server is busy (active transfer in progress)")
            else:
                print(f"  Unexpected response: {hdr['msg_type']}")

    except ConnectionRefusedError:
        print(f"  Error: No server found at {args.host}:{args.port}", file=sys.stderr)
        return 1
    except socket.timeout:
        print(f"  Error: Connection timed out", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"  Error: {exc}", file=sys.stderr)
        return 1

    return 0


def _print_status(data: dict) -> None:
    if data.get("status") == "idle":
        print("  Server status  : IDLE (no active transfer)")
        return

    print(f"  Server status  : ACTIVE")
    print(f"  Transfer ID    : {data.get('transfer_id', 'N/A')}")
    print(f"  File           : {data.get('file_name', 'N/A')}")
    print(f"  Size           : {data.get('file_size', 0) / 1e6:.2f} MB")
    pct = data.get('progress_pct', 0)
    received = data.get('chunks_received', 0)
    total    = data.get('total_chunks', 0)
    bar_len  = 36
    filled   = int(bar_len * pct / 100) if pct else 0
    bar      = "█" * filled + "░" * (bar_len - filled)
    print(f"  Progress       : [{bar}] {pct:.1f}%")
    print(f"  Chunks         : {received}/{total}")
    print(f"  Output path    : {data.get('output_path', 'N/A')}")
    print(f"  Complete       : {'Yes' if data.get('complete') else 'No'}")
    print()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="turbolane-server",
        description="TurboLane — RL-optimized parallel TCP file transfer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ── start ──────────────────────────────────────────────────────────
    p_start = sub.add_parser(
        "start",
        help="Start the TurboLane receiver server",
        description="Start the TurboLane server (receiver). Listens for incoming transfer streams.",
    )
    p_start.add_argument(
        "--host", default="0.0.0.0", metavar="HOST",
        help="Interface to bind (default: 0.0.0.0)",
    )
    p_start.add_argument(
        "--port", type=int, default=9000, metavar="PORT",
        help="TCP port to listen on (default: 9000)",
    )
    p_start.add_argument(
        "--output-dir", default="./received", metavar="DIR",
        help="Directory to save received files (default: ./received)",
    )

    # ── send ───────────────────────────────────────────────────────────
    p_send = sub.add_parser(
        "send",
        help="Send a file to a TurboLane server",
        description=(
            "Send a file using RL-optimized parallel TCP streams.\n"
            "TurboLane is embedded in the sender — no separate engine process needed."
        ),
    )
    p_send.add_argument(
        "file",
        metavar="FILE",
        help="Path to the file to send",
    )
    p_send.add_argument(
        "--host", required=True, metavar="HOST",
        help="Hostname or IP of the TurboLane server",
    )
    p_send.add_argument(
        "--port", type=int, default=9000, metavar="PORT",
        help="Server port (default: 9000)",
    )
    p_send.add_argument(
        "--streams", type=int, default=4, metavar="N",
        help="Initial number of parallel TCP streams (default: 4)",
    )
    p_send.add_argument(
        "--min-streams", type=int, default=1, metavar="N",
        help="Minimum streams TurboLane may use (default: 1)",
    )
    p_send.add_argument(
        "--max-streams", type=int, default=32, metavar="N",
        help="Maximum streams TurboLane may use (default: 32)",
    )
    p_send.add_argument(
        "--model-dir", default="models/dci", metavar="DIR",
        help="Q-table model directory (default: models/dci)",
    )
    p_send.add_argument(
        "--interval", type=float, default=5.0, metavar="SECS",
        help="TurboLane monitoring / decision interval in seconds (default: 5.0)",
    )
    p_send.add_argument(
        "--timeout", type=float, default=None, metavar="SECS",
        help="Max seconds to wait for transfer completion (default: unlimited)",
    )

    # ── status ─────────────────────────────────────────────────────────
    p_status = sub.add_parser(
        "status",
        help="Query a running TurboLane server",
        description="Connect to a TurboLane server and print its current transfer status.",
    )
    p_status.add_argument(
        "--host", default="127.0.0.1", metavar="HOST",
        help="Server hostname or IP (default: 127.0.0.1)",
    )
    p_status.add_argument(
        "--port", type=int, default=9000, metavar="PORT",
        help="Server port (default: 9000)",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    _setup_logging(args.verbose)

    dispatch = {
        "start":  cmd_start,
        "send":   cmd_send,
        "status": cmd_status,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    sys.exit(handler(args))


if __name__ == "__main__":
    main()
