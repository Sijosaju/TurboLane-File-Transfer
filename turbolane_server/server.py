"""
turbolane_server/server.py

TurboLane file transfer SERVER (receiver side).

Responsibilities:
  - Listen on a TCP port for incoming stream connections
  - Handle the HELLO / CHUNK / PING / TRANSFER_DONE protocol
  - Reassemble chunks into the output file (thread-safe, sparse write)
  - Enforce the single-transfer-at-a-time lock
  - Reject new connections with BUSY when a transfer is active
  - Expose a status endpoint (STATUS_REQ / STATUS_RESP) for the CLI

Architecture:
  - One accept-loop thread
  - One handler thread per accepted connection (one per stream)
  - Shared FileAssembler per transfer (keyed by transfer_id)
  - No TurboLane code here — server is purely the receiver/assembler

Note:
  The TurboLane RL engine lives on the SENDER side (see sender.py).
  The server (receiver) just reassembles and acks chunks as fast as possible.
"""

import os
import socket
import threading
import time
import logging
import json
import signal
import sys
from pathlib import Path
from typing import Optional

from turbolane_server.protocol import (
    MessageType,
    HEADER_SIZE,
    encode_frame,
    encode_meta,
    decode_meta,
    recv_frame,
)

logger = logging.getLogger(__name__)

# How long to wait for the next frame before timing out a stream
STREAM_TIMEOUT = 120.0

# Directory where received files are saved (overridden by CLI)
DEFAULT_OUTPUT_DIR = "./received"


# ---------------------------------------------------------------------------
# FileAssembler — thread-safe sparse file writer
# ---------------------------------------------------------------------------

class FileAssembler:
    """
    Accepts out-of-order chunk writes from multiple stream threads and
    assembles them into the correct file on disk.

    Thread-safe: multiple StreamHandler threads call write_chunk() concurrently.
    """

    def __init__(
        self,
        transfer_id: str,
        file_name: str,
        file_size: int,
        total_chunks: int,
        output_dir: str,
    ) -> None:
        self.transfer_id  = transfer_id
        self.file_name    = file_name
        self.file_size    = file_size
        self.total_chunks = total_chunks
        self.output_dir   = output_dir

        self._lock      = threading.Lock()
        self._acked     = set()
        self._complete  = threading.Event()
        self._error: Optional[str] = None

        # Pre-allocate the output file
        out_path = Path(output_dir) / file_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = str(out_path)

        with open(self.output_path, "wb") as fh:
            fh.seek(file_size - 1)
            fh.write(b"\x00")

        logger.info(
            "FileAssembler %s: pre-allocated %s (%d bytes)",
            transfer_id, self.output_path, file_size,
        )

    def write_chunk(self, chunk_idx: int, file_offset: int, data: bytes) -> bool:
        """
        Write one chunk to the pre-allocated file.
        Returns True if this was a new chunk (not a duplicate).
        """
        with self._lock:
            if chunk_idx in self._acked:
                return False   # duplicate, ignore
            # Write under lock to prevent fd seek/write races
            with open(self.output_path, "r+b") as fh:
                fh.seek(file_offset)
                fh.write(data)
            self._acked.add(chunk_idx)
            done = len(self._acked) >= self.total_chunks

        if done:
            logger.info(
                "FileAssembler %s: all %d chunks received → %s",
                self.transfer_id, self.total_chunks, self.output_path,
            )
            self._complete.set()

        return True

    def wait_complete(self, timeout: float = None) -> bool:
        return self._complete.wait(timeout=timeout)

    @property
    def is_complete(self) -> bool:
        return self._complete.is_set()

    @property
    def chunks_received(self) -> int:
        with self._lock:
            return len(self._acked)

    def get_stats(self) -> dict:
        return {
            "transfer_id":     self.transfer_id,
            "file_name":       self.file_name,
            "file_size":       self.file_size,
            "total_chunks":    self.total_chunks,
            "chunks_received": self.chunks_received,
            "progress_pct":    round(self.chunks_received / self.total_chunks * 100, 1),
            "complete":        self.is_complete,
            "output_path":     self.output_path,
        }


# ---------------------------------------------------------------------------
# StreamHandler — handles one TCP stream connection
# ---------------------------------------------------------------------------

class StreamHandler:
    """
    Handles a single TCP stream connection from a sender.

    Processes: HELLO → (CHUNK* | PING)* → TRANSFER_DONE
    Sends back: HELLO_ACK, CHUNK_ACK, PONG, COMPLETE
    """

    def __init__(
        self,
        conn: socket.socket,
        addr: tuple,
        assembler_registry: dict,   # transfer_id → FileAssembler
        registry_lock: threading.Lock,
        busy_flag: threading.Event,
        output_dir: str,
    ) -> None:
        self._conn             = conn
        self._addr             = addr
        self._assemblers       = assembler_registry
        self._registry_lock    = registry_lock
        self._busy_flag        = busy_flag
        self._output_dir       = output_dir
        self._assembler: Optional[FileAssembler] = None
        self._stream_id        = -1

    def handle(self) -> None:
        """Main handler loop for one stream connection."""
        try:
            self._conn.settimeout(STREAM_TIMEOUT)
            self._run()
        except ConnectionError as exc:
            logger.info("Stream %d closed: %s", self._stream_id, exc)
        except Exception as exc:
            logger.error("Stream %d error: %s", self._stream_id, exc, exc_info=True)
        finally:
            try:
                self._conn.close()
            except OSError:
                pass

    def _run(self) -> None:
        while True:
            hdr, payload = recv_frame(self._conn)
            msg = hdr["msg_type"]

            if msg == MessageType.HELLO:
                self._handle_hello(hdr, payload)

            elif msg == MessageType.CHUNK:
                self._handle_chunk(hdr, payload)

            elif msg == MessageType.PING:
                self._handle_ping(hdr)

            elif msg == MessageType.TRANSFER_DONE:
                self._handle_done()
                break

            elif msg == MessageType.STATUS_REQ:
                self._handle_status()

            else:
                logger.warning("Stream %d: unexpected msg %s", self._stream_id, msg)

    def _handle_hello(self, hdr: dict, payload: bytes) -> None:
        meta = decode_meta(payload)
        transfer_id  = meta["transfer_id"]
        stream_id    = meta["stream_id"]
        file_name    = meta["file_name"]
        file_size    = meta["file_size"]
        total_chunks = meta["total_chunks"]
        self._stream_id = stream_id

        with self._registry_lock:
            if transfer_id not in self._assemblers:
                # First stream for this transfer — check busy
                if self._busy_flag.is_set():
                    busy_resp = encode_frame(
                        msg_type=MessageType.BUSY,
                        payload=encode_meta({"reason": "A transfer is already in progress"}),
                    )
                    self._conn.sendall(busy_resp)
                    logger.warning("Rejected new transfer — server busy")
                    return

                # Register new assembler and set busy
                assembler = FileAssembler(
                    transfer_id=transfer_id,
                    file_name=file_name,
                    file_size=file_size,
                    total_chunks=total_chunks,
                    output_dir=self._output_dir,
                )
                self._assemblers[transfer_id] = assembler
                self._busy_flag.set()
                logger.info(
                    "New transfer %s: %s (%d bytes, %d chunks)",
                    transfer_id, file_name, file_size, total_chunks,
                )
            else:
                assembler = self._assemblers[transfer_id]

        self._assembler = assembler

        ack = encode_frame(
            msg_type=MessageType.HELLO_ACK,
            stream_id=stream_id,
        )
        self._conn.sendall(ack)
        logger.info("Stream %d: HELLO_ACK sent for transfer %s", stream_id, transfer_id)

    def _handle_chunk(self, hdr: dict, payload: bytes) -> None:
        if self._assembler is None:
            logger.error("Stream %d: CHUNK received before HELLO", self._stream_id)
            return

        chunk_idx   = hdr["chunk_idx"]
        file_offset = hdr["file_offset"]

        self._assembler.write_chunk(chunk_idx, file_offset, payload)

        ack = encode_frame(
            msg_type=MessageType.CHUNK_ACK,
            stream_id=self._stream_id,
            chunk_idx=chunk_idx,
        )
        self._conn.sendall(ack)

    def _handle_ping(self, hdr: dict) -> None:
        pong = encode_frame(
            msg_type=MessageType.PONG,
            stream_id=self._stream_id,
        )
        self._conn.sendall(pong)
        logger.debug("Stream %d: PONG sent", self._stream_id)

    def _handle_done(self) -> None:
        if self._assembler is None:
            return
        complete = encode_frame(
            msg_type=MessageType.COMPLETE,
            stream_id=self._stream_id,
            payload=encode_meta({"output_path": self._assembler.output_path}),
        )
        self._conn.sendall(complete)
        logger.info("Stream %d: TRANSFER_DONE acknowledged", self._stream_id)

    def _handle_status(self) -> None:
        if self._assembler:
            stats = self._assembler.get_stats()
        else:
            stats = {"status": "idle"}
        resp = encode_frame(
            msg_type=MessageType.STATUS_RESP,
            payload=encode_meta(stats),
        )
        self._conn.sendall(resp)


# ---------------------------------------------------------------------------
# TurboLaneServer — main server class
# ---------------------------------------------------------------------------

class TurboLaneServer:
    """
    TCP server that listens for incoming file transfer streams.

    Usage:
        server = TurboLaneServer(host="0.0.0.0", port=9000, output_dir="./received")
        server.start()          # blocks (runs the accept loop)
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9000,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        backlog: int = 64,
    ) -> None:
        self.host       = host
        self.port       = port
        self.output_dir = output_dir
        self.backlog    = backlog

        # Single-transfer enforcement
        self._busy_flag = threading.Event()

        # Active assemblers: transfer_id → FileAssembler
        self._assemblers: dict = {}
        self._registry_lock = threading.Lock()

        # Cleanup thread watches for completed transfers
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()

        self._sock: Optional[socket.socket] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, ready_event: Optional[threading.Event] = None) -> None:
        """
        Start listening. Blocks until shutdown signal received.

        Args:
            ready_event: If provided, set() once the socket is bound and
                         listening (useful for tests / programmatic callers).
        """
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(self.backlog)
        self._sock.settimeout(1.0)  # so accept() can be interrupted

        # Start cleanup watcher
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="transfer-cleanup",
        )
        self._cleanup_thread.start()

        # Install signal handlers only when running on the main thread
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT,  self._on_signal)
            signal.signal(signal.SIGTERM, self._on_signal)

        print(f"\n  TurboLane Server listening on {self.host}:{self.port}")
        print(f"  Output directory : {os.path.abspath(self.output_dir)}")
        print(f"  Press Ctrl-C to stop\n")
        logger.info("Server started on %s:%d", self.host, self.port)

        # Signal that we are ready to accept connections
        if ready_event is not None:
            ready_event.set()

        self._accept_loop()

    def _accept_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                conn, addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            logger.info("New connection from %s:%d", *addr)
            handler = StreamHandler(
                conn=conn,
                addr=addr,
                assembler_registry=self._assemblers,
                registry_lock=self._registry_lock,
                busy_flag=self._busy_flag,
                output_dir=self.output_dir,
            )
            t = threading.Thread(
                target=handler.handle,
                name=f"handler-{addr[0]}-{addr[1]}",
                daemon=True,
            )
            t.start()

        logger.info("Accept loop exited")

    def _cleanup_loop(self) -> None:
        """Periodically checks for completed transfers and clears the busy flag."""
        while not self._shutdown.is_set():
            time.sleep(2.0)
            with self._registry_lock:
                completed = [
                    tid for tid, asm in self._assemblers.items()
                    if asm.is_complete
                ]
                for tid in completed:
                    asm = self._assemblers.pop(tid)
                    logger.info(
                        "Transfer %s complete — %s",
                        tid, asm.output_path,
                    )
                    print(
                        f"\n  ✓ Transfer complete: {asm.file_name}"
                        f"  ({asm.file_size/1e6:.1f} MB) → {asm.output_path}\n"
                    )

                if completed and not self._assemblers:
                    self._busy_flag.clear()
                    logger.info("Server: ready for next transfer")
                    print("  Server: ready for next transfer\n")

    def _on_signal(self, signum, frame) -> None:
        print("\n  Shutting down server...")
        self._shutdown.set()
        if self._sock:
            self._sock.close()
        sys.exit(0)
