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
import queue
import signal
import sys
from pathlib import Path
from typing import Optional

from turbolane_server.protocol import (
    MessageType,
    encode_frame,
    encode_meta,
    decode_meta,
    recv_frame,
)

logger = logging.getLogger(__name__)

# How long to wait for the next frame before timing out a stream
STREAM_TIMEOUT = 120.0

# If no activity is observed for this long, abandon the partial transfer.
STALE_TRANSFER_TIMEOUT = 180.0

# Number of chunks buffered for async disk writer (0 = unbounded).
WRITE_QUEUE_MAX = 0

# Directory where received files are saved (overridden by CLI)
DEFAULT_OUTPUT_DIR = "./received"


def _sanitize_file_name(raw_name: str) -> str:
    """
    Keep only the leaf filename to prevent directory traversal.
    """
    name = Path(str(raw_name)).name
    if not name or name in {".", ".."}:
        raise ValueError("Invalid file_name metadata")
    return name


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
        self._acked     = set()      # chunks durably written
        self._seen      = set()      # chunks accepted (queued or written)
        self._complete  = threading.Event()
        self._error: Optional[str] = None
        self._last_activity = time.monotonic()
        self._closed = False

        # Pre-allocate the output file
        out_path = Path(output_dir) / file_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = str(out_path)

        with open(self.output_path, "wb") as fh:
            if file_size > 0:
                fh.seek(file_size - 1)
                fh.write(b"\x00")

        # Keep one fd open for the full transfer to avoid per-chunk open/close.
        self._fh = open(self.output_path, "r+b")

        # Writer thread decouples ACK timing from disk latency.
        self._write_queue: queue.Queue = queue.Queue(maxsize=WRITE_QUEUE_MAX)
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name=f"writer-{transfer_id}",
            daemon=True,
        )
        self._writer_thread.start()

        if self.total_chunks == 0:
            self._complete.set()
            logger.info("FileAssembler %s: zero-byte transfer", transfer_id)

        logger.info(
            "FileAssembler %s: pre-allocated %s (%d bytes)",
            transfer_id, self.output_path, file_size,
        )

    def write_chunk(self, chunk_idx: int, file_offset: int, data: bytes) -> bool:
        """
        Queue one chunk for async write.
        Returns True if this was a new chunk (not a duplicate).
        """
        with self._lock:
            self._last_activity = time.monotonic()
            if self._closed:
                return False
            if chunk_idx in self._seen:
                return False
            self._seen.add(chunk_idx)

        self._write_queue.put((chunk_idx, file_offset, data))
        return True

    def _writer_loop(self) -> None:
        while True:
            item = self._write_queue.get()
            if item is None:
                self._write_queue.task_done()
                break

            chunk_idx, file_offset, data = item
            try:
                self._fh.seek(file_offset)
                self._fh.write(data)

                with self._lock:
                    self._acked.add(chunk_idx)
                    self._last_activity = time.monotonic()
                    done = len(self._acked) >= self.total_chunks

                if done:
                    logger.info(
                        "FileAssembler %s: all %d chunks received -> %s",
                        self.transfer_id,
                        self.total_chunks,
                        self.output_path,
                    )
                    self._complete.set()

            except Exception as exc:
                with self._lock:
                    self._error = str(exc)
                logger.error(
                    "FileAssembler %s write error on chunk %d: %s",
                    self.transfer_id,
                    chunk_idx,
                    exc,
                )
            finally:
                self._write_queue.task_done()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True

        self._write_queue.put(None)
        self._writer_thread.join(timeout=5.0)

        if self._writer_thread.is_alive():
            logger.warning(
                "FileAssembler %s: writer thread did not stop in time",
                self.transfer_id,
            )
            return

        try:
            self._fh.close()
        except OSError:
            pass

    def touch(self) -> None:
        with self._lock:
            self._last_activity = time.monotonic()

    def is_stale(self, timeout_s: float) -> bool:
        with self._lock:
            if self._complete.is_set():
                return False
            return (time.monotonic() - self._last_activity) > timeout_s

    def wait_complete(self, timeout: float = None) -> bool:
        return self._complete.wait(timeout=timeout)

    @property
    def is_complete(self) -> bool:
        return self._complete.is_set()

    @property
    def chunks_received(self) -> int:
        with self._lock:
            return len(self._acked)

    @property
    def error(self) -> Optional[str]:
        with self._lock:
            return self._error

    def get_stats(self) -> dict:
        progress_pct = (
            round(self.chunks_received / self.total_chunks * 100, 1)
            if self.total_chunks > 0
            else (100.0 if self.is_complete else 0.0)
        )
        return {
            "transfer_id":     self.transfer_id,
            "file_name":       self.file_name,
            "file_size":       self.file_size,
            "total_chunks":    self.total_chunks,
            "chunks_received": self.chunks_received,
            "progress_pct":    progress_pct,
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
        raw_file_name = meta["file_name"]
        file_size    = meta["file_size"]
        total_chunks = meta["total_chunks"]
        self._stream_id = stream_id

        try:
            file_name = _sanitize_file_name(raw_file_name)
        except ValueError as exc:
            logger.warning("Stream %d: invalid file_name '%s': %s", stream_id, raw_file_name, exc)
            err = encode_frame(
                msg_type=MessageType.ERROR,
                stream_id=stream_id,
                payload=encode_meta({"reason": "Invalid file_name metadata"}),
            )
            self._conn.sendall(err)
            return
        if file_name != raw_file_name:
            logger.warning(
                "Stream %d: normalized file_name '%s' -> '%s'",
                stream_id,
                raw_file_name,
                file_name,
            )

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
                assembler.touch()

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
        if self._assembler is not None:
            self._assembler.touch()
        pong = encode_frame(
            msg_type=MessageType.PONG,
            stream_id=self._stream_id,
        )
        self._conn.sendall(pong)
        logger.debug("Stream %d: PONG sent", self._stream_id)

    def _handle_done(self) -> None:
        if self._assembler is None:
            return
        self._assembler.touch()

        # COMPLETE means durable assembly is done, not just queued.
        if not self._assembler.wait_complete(timeout=STREAM_TIMEOUT):
            err = encode_frame(
                msg_type=MessageType.ERROR,
                stream_id=self._stream_id,
                payload=encode_meta({"reason": "Timed out waiting for file assembly"}),
            )
            self._conn.sendall(err)
            logger.warning("Stream %d: TRANSFER_DONE before assembly completion", self._stream_id)
            return

        if self._assembler.error:
            err = encode_frame(
                msg_type=MessageType.ERROR,
                stream_id=self._stream_id,
                payload=encode_meta({"reason": self._assembler.error}),
            )
            self._conn.sendall(err)
            logger.warning(
                "Stream %d: assembly error before COMPLETE: %s",
                self._stream_id,
                self._assembler.error,
            )
            return

        complete = encode_frame(
            msg_type=MessageType.COMPLETE,
            stream_id=self._stream_id,
            payload=encode_meta({"output_path": self._assembler.output_path}),
        )
        self._conn.sendall(complete)
        logger.info("Stream %d: TRANSFER_DONE acknowledged", self._stream_id)

    def _handle_status(self) -> None:
        with self._registry_lock:
            active = next(iter(self._assemblers.values()), None)

        if active is None:
            stats = {"status": "idle"}
        else:
            stats = active.get_stats()
            stats["status"] = "active"

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
                stale = [
                    tid for tid, asm in self._assemblers.items()
                    if asm.is_stale(STALE_TRANSFER_TIMEOUT)
                ]
                for tid in completed:
                    asm = self._assemblers.pop(tid)
                    asm.close()
                    logger.info(
                        "Transfer %s complete — %s",
                        tid, asm.output_path,
                    )
                    print(
                        f"\n  ✓ Transfer complete: {asm.file_name}"
                        f"  ({asm.file_size/1e6:.1f} MB) → {asm.output_path}\n"
                    )

                for tid in stale:
                    asm = self._assemblers.pop(tid)
                    asm.close()
                    logger.warning(
                        "Transfer %s timed out after %.0fs without activity; marking failed",
                        tid,
                        STALE_TRANSFER_TIMEOUT,
                    )
                    print(
                        f"\n  ! Transfer aborted (stale): {asm.file_name}"
                        f"  ({asm.file_size/1e6:.1f} MB)\n"
                    )

                if self._busy_flag.is_set() and not self._assemblers:
                    self._busy_flag.clear()
                    logger.info("Server: ready for next transfer")
                    print("  Server: ready for next transfer\n")

    def _on_signal(self, signum, frame) -> None:
        print("\n  Shutting down server...")
        self._shutdown.set()
        if self._sock:
            self._sock.close()
        sys.exit(0)

