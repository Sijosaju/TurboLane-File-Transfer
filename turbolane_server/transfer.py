"""
turbolane_server/transfer.py

TransferSession — orchestrates a single file transfer on the sender side.

Responsibilities:
  - Split the file into fixed-size chunks
  - Assign chunks to N parallel streams (round-robin)
  - Spawn one worker thread per stream; each worker connects to the receiver
  - Handle PING→PONG RTT measurement per stream
  - Receive CHUNK_ACKs and track completion
  - Expose adjust_streams(n) so the TurboLane adapter can change
    the parallelism mid-transfer (streams can be added or removed live)
  - Report progress to the terminal

No TurboLane code lives here. No engine imports. Pure transfer logic.
"""

import os
import math
import socket
import struct
import threading
import time
import logging
import queue
from pathlib import Path
from typing import Optional

from turbolane_server.protocol import (
    MessageType,
    CHUNK_SIZE,
    HEADER_SIZE,
    encode_frame,
    encode_meta,
    decode_meta,
    recv_frame,
)
from turbolane_server.metrics import MetricsCollector, StreamMetrics

logger = logging.getLogger(__name__)

# How long a worker waits for an ACK before considering a chunk lost
# FIX #1: Increased from 10.0 → 30.0 to tolerate real Wi-Fi LAN latency.
# Over localhost this was fine; over Wi-Fi with 1 MB chunks and TCP congestion
# 10s was too tight and caused spurious timeouts + stream churn.
ACK_TIMEOUT = 30.0          # seconds (was 10.0)

PING_INTERVAL = 4.0         # seconds between RTT probes per stream

# FIX #2: Increased from 10.0 → 20.0 to give the receiver's TCP stack more
# time to accept connections when many streams are being spawned rapidly.
CONNECT_TIMEOUT = 20.0      # seconds (was 10.0)

MAX_CHUNK_RETRIES = 3       # retransmit attempts before giving up


class ChunkQueue:
    """
    Thread-safe queue of (chunk_idx, file_offset, length) tuples.
    Workers pull from here; the queue is pre-filled before transfer starts.
    """

    def __init__(self, file_path: str, chunk_size: int = CHUNK_SIZE) -> None:
        self.file_path  = file_path
        self.chunk_size = chunk_size
        self.file_size  = os.path.getsize(file_path)
        self.total_chunks = math.ceil(self.file_size / chunk_size)

        self._q: queue.Queue = queue.Queue()
        self._completed = threading.Event()
        self._lock = threading.Lock()
        self._sent_count = 0
        self._acked_count = 0

        # Pre-fill the queue
        for idx in range(self.total_chunks):
            offset = idx * chunk_size
            length = min(chunk_size, self.file_size - offset)
            self._q.put((idx, offset, length))

    def get(self, timeout: float = 1.0) -> Optional[tuple]:
        """Pop the next (chunk_idx, offset, length) or None if empty/timed out."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def requeue(self, chunk_idx: int, offset: int, length: int) -> None:
        """Put a failed chunk back for retry."""
        self._q.put((chunk_idx, offset, length))

    def mark_acked(self) -> None:
        with self._lock:
            self._acked_count += 1
            if self._acked_count >= self.total_chunks:
                self._completed.set()

    def mark_sent(self) -> None:
        with self._lock:
            self._sent_count += 1

    def wait_complete(self, timeout: float = None) -> bool:
        return self._completed.wait(timeout=timeout)

    @property
    def is_done(self) -> bool:
        return self._completed.is_set()

    @property
    def progress(self) -> tuple[int, int]:
        with self._lock:
            return self._acked_count, self.total_chunks


class StreamWorker:
    """
    One parallel TCP stream connecting sender → receiver.

    Lifecycle:
        worker = StreamWorker(stream_id, host, port, chunk_queue, metrics_sm)
        worker.start()
        ...
        worker.stop()   # graceful shutdown; unfinished chunks go back to queue
    """

    def __init__(
        self,
        stream_id: int,
        host: str,
        port: int,
        chunk_queue: ChunkQueue,
        stream_metrics: StreamMetrics,
        transfer_id: str,
        file_name: str,
        total_chunks: int,
        file_size: int,
    ) -> None:
        self.stream_id      = stream_id
        self.host           = host
        self.port           = port
        self._cq            = chunk_queue
        self._sm            = stream_metrics
        self._transfer_id   = transfer_id
        self._file_name     = file_name
        self._total_chunks  = total_chunks
        self._file_size     = file_size

        self._stop_event    = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None
        self._last_ping_time: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """
        Connect to receiver and start the worker thread.
        Returns True on successful connection, False on failure.
        """
        try:
            self._sock = socket.create_connection(
                (self.host, self.port), timeout=CONNECT_TIMEOUT
            )
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock.settimeout(ACK_TIMEOUT)
        except OSError as exc:
            logger.error("Stream %d: connect failed: %s", self.stream_id, exc)
            return False

        # Send HELLO on this stream
        if not self._send_hello():
            self._sock.close()
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"stream-{self.stream_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info("Stream %d: started → %s:%d", self.stream_id, self.host, self.port)
        return True

    def stop(self) -> None:
        """Signal this stream to stop after finishing its current chunk."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=ACK_TIMEOUT + 2)
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
        logger.info("Stream %d: stopped", self.stream_id)

    @property
    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ------------------------------------------------------------------
    # Internal: HELLO handshake
    # ------------------------------------------------------------------

    def _send_hello(self) -> bool:
        meta = {
            "transfer_id":  self._transfer_id,
            "stream_id":    self.stream_id,
            "file_name":    self._file_name,
            "file_size":    self._file_size,
            "total_chunks": self._total_chunks,
            "chunk_size":   CHUNK_SIZE,
        }
        frame = encode_frame(
            msg_type=MessageType.HELLO,
            stream_id=self.stream_id,
            payload=encode_meta(meta),
        )
        try:
            self._sock.sendall(frame)
            hdr, payload = recv_frame(self._sock)
            if hdr["msg_type"] == MessageType.BUSY:
                info = decode_meta(payload) if payload else {}
                logger.error("Stream %d: server BUSY — %s", self.stream_id, info.get("reason", ""))
                return False
            if hdr["msg_type"] != MessageType.HELLO_ACK:
                logger.error("Stream %d: unexpected HELLO response: %s", self.stream_id, hdr["msg_type"])
                return False
            return True
        except Exception as exc:
            logger.error("Stream %d: HELLO failed: %s", self.stream_id, exc)
            return False

    # ------------------------------------------------------------------
    # Internal: main worker loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """
        Worker loop:
          1. Pull a chunk from the queue
          2. Read it from the file
          3. Send CHUNK frame
          4. Wait for CHUNK_ACK (with retries)
          5. Periodically send PING for RTT measurement
        """
        with open(self._cq.file_path, "rb") as fh:
            while not self._stop_event.is_set() and not self._cq.is_done:

                # Periodic RTT ping
                self._maybe_ping()

                # Get next chunk
                item = self._cq.get(timeout=0.5)
                if item is None:
                    # Queue empty — either done or being refilled by retries
                    continue

                chunk_idx, offset, length = item
                success = False

                for attempt in range(MAX_CHUNK_RETRIES):
                    try:
                        fh.seek(offset)
                        data = fh.read(length)

                        frame = encode_frame(
                            msg_type=MessageType.CHUNK,
                            stream_id=self.stream_id,
                            chunk_idx=chunk_idx,
                            total_chunks=self._total_chunks,
                            seq=chunk_idx,
                            file_offset=offset,
                            payload=data,
                        )

                        send_start = time.monotonic()
                        self._sock.sendall(frame)
                        self._sm.record_bytes(len(data))
                        self._cq.mark_sent()

                        # Wait for ACK — tolerate interleaved PONG frames
                        while True:
                            hdr, _payload = recv_frame(self._sock)
                            if hdr["msg_type"] == MessageType.PONG:
                                # RTT probe response — record it and keep waiting for ACK
                                self._sm.record_pong()
                                logger.debug("Stream %d: PONG received (interleaved)", self.stream_id)
                                continue
                            break   # got a real response

                        if hdr["msg_type"] == MessageType.CHUNK_ACK and hdr["chunk_idx"] == chunk_idx:
                            self._cq.mark_acked()
                            success = True
                            logger.debug(
                                "Stream %d chunk %d ACKed in %.1f ms",
                                self.stream_id, chunk_idx,
                                (time.monotonic() - send_start) * 1000,
                            )
                            break
                        else:
                            logger.warning(
                                "Stream %d chunk %d: unexpected response %s (attempt %d)",
                                self.stream_id, chunk_idx, hdr["msg_type"], attempt + 1,
                            )
                            self._sm.record_nack()

                    except socket.timeout:
                        logger.warning(
                            "Stream %d chunk %d: ACK timeout (attempt %d/%d)",
                            self.stream_id, chunk_idx, attempt + 1, MAX_CHUNK_RETRIES,
                        )
                        self._sm.record_nack()
                    except (ConnectionError, OSError) as exc:
                        logger.error("Stream %d: socket error: %s", self.stream_id, exc)
                        self._stop_event.set()
                        break

                if not success:
                    if not self._stop_event.is_set():
                        logger.error(
                            "Stream %d chunk %d: failed after %d attempts, requeueing",
                            self.stream_id, chunk_idx, MAX_CHUNK_RETRIES,
                        )
                        self._cq.requeue(chunk_idx, offset, length)

        logger.info("Stream %d: worker loop exited", self.stream_id)

    # ------------------------------------------------------------------
    # Internal: RTT probe
    # ------------------------------------------------------------------

    def _maybe_ping(self) -> None:
        now = time.monotonic()
        if now - self._last_ping_time < PING_INTERVAL:
            return
        self._last_ping_time = now
        try:
            frame = encode_frame(
                msg_type=MessageType.PING,
                stream_id=self.stream_id,
            )
            self._sm.record_ping_sent()
            self._sock.sendall(frame)
            # We don't block on PONG here — it arrives interleaved with ACKs
            # and is handled by a lightweight non-blocking check below.
            # For simplicity in this design, PONG is handled by the receiver
            # echoing immediately and the ACK receiver picking it up as an
            # interleaved message. In this single-threaded per-stream design
            # we process it whenever it arrives next.
        except OSError:
            pass


class TransferSession:
    """
    Manages the complete lifecycle of one outbound file transfer.

    Usage:
        session = TransferSession(
            file_path="/data/dataset.tar",
            receiver_host="192.168.1.50",
            receiver_port=9001,
            num_streams=4,
            metrics_collector=mc,
        )
        session.start()
        session.wait()         # blocks until transfer complete or error
        session.get_stats()
    """

    def __init__(
        self,
        file_path: str,
        receiver_host: str,
        receiver_port: int,
        num_streams: int,
        metrics_collector: MetricsCollector,
        transfer_id: Optional[str] = None,
    ) -> None:
        self.file_path       = file_path
        self.receiver_host   = receiver_host
        self.receiver_port   = receiver_port
        self.metrics         = metrics_collector

        self._file_name   = Path(file_path).name
        self._file_size   = os.path.getsize(file_path)
        self._transfer_id = transfer_id or f"xfer-{int(time.time())}"

        self._chunk_queue = ChunkQueue(file_path)
        self._workers: dict[int, StreamWorker] = {}
        self._lock = threading.Lock()
        self._target_streams = num_streams
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._done_event = threading.Event()
        self._error: Optional[str] = None

        # FIX #3: Monotonically increasing stream ID counter.
        # Previously, _spawn_stream(stream_id) reused IDs after streams were
        # stopped, causing a race where two workers briefly shared the same ID
        # and self._workers[stream_id] was silently overwritten. Now every
        # spawned stream gets a unique ID regardless of how many have been
        # stopped and restarted.
        self._next_stream_id = 0

        # Progress monitor thread
        self._monitor_thread: Optional[threading.Thread] = None

        logger.info(
            "TransferSession %s: file=%s size=%d streams=%d dest=%s:%d",
            self._transfer_id, self._file_name, self._file_size,
            num_streams, receiver_host, receiver_port,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn initial worker streams and begin transfer."""
        self._start_time = time.monotonic()
        for _ in range(self._target_streams):
            self._spawn_stream()

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="transfer-monitor",
            daemon=True,
        )
        self._monitor_thread.start()

    def wait(self, timeout: float = None) -> bool:
        """
        Block until transfer completes or timeout expires.
        Returns True if transfer completed successfully.
        """
        completed = self._done_event.wait(timeout=timeout)
        return completed and self._error is None

    def adjust_streams(self, new_count: int) -> None:
        """
        Called by the TurboLane adapter when the engine recommends
        a different number of parallel streams.

        Safely adds or removes workers without interrupting active transfers.
        """
        with self._lock:
            current = len(self._workers)
            self._target_streams = new_count

        if new_count > current:
            for _ in range(new_count - current):
                logger.info("Adapter: adding stream (total → %d)", new_count)
                self._spawn_stream()

        elif new_count < current:
            # Stop excess streams (highest IDs first, least likely to have work)
            with self._lock:
                ids_to_stop = sorted(self._workers.keys(), reverse=True)
                ids_to_stop = ids_to_stop[:current - new_count]

            for sid in ids_to_stop:
                logger.info("Adapter: removing stream %d (total → %d)", sid, new_count)
                self._stop_stream(sid)

    def get_stats(self) -> dict:
        elapsed = (
            (self._end_time or time.monotonic()) - self._start_time
            if self._start_time else 0.0
        )
        acked, total = self._chunk_queue.progress
        throughput = (
            (self._file_size * 8) / (elapsed * 1e6)
            if elapsed > 0 else 0.0
        )
        return {
            "transfer_id":   self._transfer_id,
            "file_name":     self._file_name,
            "file_size":     self._file_size,
            "total_chunks":  total,
            "acked_chunks":  acked,
            "progress_pct":  round(acked / total * 100, 1) if total else 0.0,
            "elapsed_s":     round(elapsed, 2),
            "throughput_mbps": round(throughput, 2),
            "active_streams": len(self._workers),
            "completed":     self._done_event.is_set(),
            "error":         self._error,
        }

    def abort(self) -> None:
        """Stop all streams immediately."""
        with self._lock:
            ids = list(self._workers.keys())
        for sid in ids:
            self._stop_stream(sid)
        self._error = "aborted"
        self._done_event.set()

    # ------------------------------------------------------------------
    # Internal: stream management
    # ------------------------------------------------------------------

    def _spawn_stream(self) -> None:
        # FIX #3: Always use a fresh, never-reused stream ID.
        with self._lock:
            stream_id = self._next_stream_id
            self._next_stream_id += 1

        sm = self.metrics.register_stream(stream_id)
        worker = StreamWorker(
            stream_id=stream_id,
            host=self.receiver_host,
            port=self.receiver_port,
            chunk_queue=self._chunk_queue,
            stream_metrics=sm,
            transfer_id=self._transfer_id,
            file_name=self._file_name,
            total_chunks=self._chunk_queue.total_chunks,
            file_size=self._file_size,
        )
        ok = worker.start()
        if ok:
            with self._lock:
                self._workers[stream_id] = worker
        else:
            self.metrics.unregister_stream(stream_id)
            if not self._workers:
                self._error = f"All streams failed to connect"
                self._done_event.set()

    def _stop_stream(self, stream_id: int) -> None:
        with self._lock:
            worker = self._workers.pop(stream_id, None)
        if worker:
            threading.Thread(
                target=worker.stop,
                daemon=True,
                name=f"stream-stop-{stream_id}",
            ).start()
            self.metrics.unregister_stream(stream_id)

    # ------------------------------------------------------------------
    # Internal: progress monitor
    # ------------------------------------------------------------------

    def _monitor_loop(self) -> None:
        """Watches transfer progress and signals completion."""
        while not self._done_event.is_set():
            acked, total = self._chunk_queue.progress

            # Print progress bar
            pct = acked / total if total else 0
            bar_len = 40
            filled = int(bar_len * pct)
            bar = "█" * filled + "░" * (bar_len - filled)
            elapsed = time.monotonic() - self._start_time
            mbps = (acked * CHUNK_SIZE * 8) / (elapsed * 1e6) if elapsed > 0 else 0

            print(
                f"\r  [{bar}] {pct*100:5.1f}%  {acked}/{total} chunks  "
                f"{mbps:.1f} Mbps  {len(self._workers)} streams  ",
                end="",
                flush=True,
            )

            if self._chunk_queue.wait_complete(timeout=1.0):
                self._end_time = time.monotonic()
                print()  # newline after progress bar
                logger.info(
                    "Transfer %s complete in %.2f s  (%.2f Mbps)",
                    self._transfer_id,
                    self._end_time - self._start_time,
                    (self._file_size * 8) / ((self._end_time - self._start_time) * 1e6),
                )
                # Stop all streams cleanly
                with self._lock:
                    ids = list(self._workers.keys())
                for sid in ids:
                    self._stop_stream(sid)
                self.metrics.unregister_all()
                self._done_event.set()
                break

            # Check for zombie workers (all crashed, transfer stalled)
            with self._lock:
                alive = [w for w in self._workers.values() if w.is_alive]
            if not alive and not self._chunk_queue.is_done:
                logger.error("All stream workers died — transfer failed")
                self._error = "All stream workers died"
                self._done_event.set()
                break