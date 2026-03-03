#!/usr/bin/env python3
"""
External fixed-stream sender for TurboLane benchmarks.

Purpose:
  Run sender-side transfer with a fixed number of parallel streams while
  completely bypassing TurboLaneAdapter / TurboLaneEngine / RL components.

This script is intended for baseline experiments only.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path


# Allow running directly from repo without requiring editable install.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turbolane_server.metrics import MetricsCollector
from turbolane_server.transfer import TransferSession


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fixed-sender",
        description=(
            "TurboLane external fixed-stream sender baseline "
            "(no RL/adapter/control loop)."
        ),
    )
    parser.add_argument("file", help="Path to file to send")
    parser.add_argument("--host", required=True, help="Receiver hostname or IP")
    parser.add_argument("--port", type=int, default=9000, help="Receiver port")
    parser.add_argument(
        "--streams",
        type=_positive_int,
        required=True,
        metavar="N",
        help="Fixed number of parallel TCP streams",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="SECS",
        help="Max seconds to wait for transfer completion (default: unlimited)",
    )
    parser.add_argument(
        "--transfer-id",
        default=None,
        help="Optional explicit transfer_id (default: auto-generated)",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write result JSON for experiment automation",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def _print_start(file_path: str, host: str, port: int, streams: int) -> None:
    file_size = os.path.getsize(file_path)
    print("\n  TurboLane Fixed Sender (External Baseline)")
    print(f"  File     : {os.path.abspath(file_path)}")
    print(f"  Size     : {file_size / 1e6:.2f} MB")
    print(f"  Receiver : {host}:{port}")
    print(f"  Streams  : {streams} fixed (RL disabled path)\n")


def _print_summary(result: dict) -> None:
    status = "COMPLETED" if result["success"] else "FAILED"
    error = f" - {result['error']}" if result.get("error") else ""

    print(f"\n  {'─' * 56}")
    print("  Fixed Baseline Summary")
    print(f"  {'─' * 56}")
    print(f"  Status          : {status}{error}")
    print(f"  File            : {result['file_name']}")
    print(f"  Size            : {result['file_size'] / 1e6:.2f} MB")
    print(f"  Duration        : {result['elapsed_s']:.2f} s")
    print(f"  Avg throughput  : {result['throughput_mbps']:.2f} Mbps")
    print(f"  Chunks          : {result['acked_chunks']}/{result['total_chunks']}")
    print(f"  Progress        : {result['progress_pct']}%")
    print(f"  Active streams  : {result['active_streams']}")
    print(f"  Fixed streams   : {result['fixed_streams']}")
    print(f"  {'─' * 56}\n")


def run_fixed_sender(args: argparse.Namespace) -> int:
    if not os.path.isfile(args.file):
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        return 1

    _print_start(args.file, args.host, args.port, args.streams)

    metrics = MetricsCollector()
    session = TransferSession(
        file_path=args.file,
        receiver_host=args.host,
        receiver_port=args.port,
        num_streams=args.streams,
        metrics_collector=metrics,
        transfer_id=args.transfer_id,
    )

    started_at = time.monotonic()
    success = False

    try:
        session.start()
        success = session.wait(timeout=args.timeout)
    except KeyboardInterrupt:
        print("\n  Interrupted: aborting transfer...")
        session.abort()
        success = False

    ended_at = time.monotonic()
    stats = session.get_stats()
    stats["success"] = success
    stats["fixed_streams"] = args.streams
    stats["wall_time_s"] = round(ended_at - started_at, 3)

    _print_summary(stats)

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)
        print(f"  Result JSON    : {output_path.resolve()}")

    return 0 if success else 1


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)
    raise SystemExit(run_fixed_sender(args))


if __name__ == "__main__":
    main()
