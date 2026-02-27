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
import threading
import time
import logging
import queue
from pathlib import Path
from typing import Optional

from turbolane_server.protocol import (
    MessageType,
    CHUNK_SIZE,
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

# Number of in-flight chunks per stream. >1 removes stop-and-wait RTT stalls.
PIPELINE_WINDOW = 8

# Short poll timeout so the sender can process ACK/PONG frames continuously.
ACK_POLL_TIMEOUT = 0.2

# Reduce terminal I/O overhead from frequent progress updates.
PROGRESS_UPDATE_INTERVAL = 2.0


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

        # Zero-byte transfers are complete as soon as metadata is exchanged.
        if self.total_chunks == 0:
            self._completed.set()

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

    def send_transfer_done(self) -> None:
        """
        FIX (Bug 1): Send TRANSFER_DONE frame and wait for COMPLETE response
        before the socket is closed. Previously the socket was closed abruptly
        which caused WinError 10054/10053 (connection reset/aborted) on the
        server side as it was still blocked on recv_frame() waiting for the
        next message.
        """
        if self._sock is None:
            return
        prev_timeout = self._sock.gettimeout()
        try:
            self._sock.settimeout(ACK_TIMEOUT)
            frame = encode_frame(
                msg_type=MessageType.TRANSFER_DONE,
                stream_id=self.stream_id,
            )
            self._sock.sendall(frame)
            # Drain responses until we get COMPLETE (ignore interleaved PONGs)
            while True:
                hdr, _payload = recv_frame(self._sock)
                if hdr["msg_type"] == MessageType.PONG:
                    continue
                if hdr["msg_type"] == MessageType.COMPLETE:
                    logger.info("Stream %d: COMPLETE received", self.stream_id)
                break
        except OSError as exc:
            logger.warning("Stream %d: send_transfer_done error: %s", self.stream_id, exc)
        finally:
            try:
                self._sock.settimeout(prev_timeout)
            except OSError:
                pass

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
        Worker loop with pipelining:
          1. Keep up to PIPELINE_WINDOW chunks in-flight per stream
          2. Continuously process CHUNK_ACK / PONG frames
          3. Retry only timed-out in-flight chunks
        """
        inflight: dict[int, dict] = {}

        try:
            self._sock.settimeout(ACK_POLL_TIMEOUT)
        except OSError:
            pass

        try:
            with open(self._cq.file_path, "rb") as fh:
                while not self._stop_event.is_set() and not self._cq.is_done:
                    self._maybe_ping()
                    self._fill_window(fh, inflight)
                    self._drain_frames(inflight)
                    self._retry_timeouts(inflight)

                    if not inflight:
                        time.sleep(0.01)

                while (
                    not self._stop_event.is_set()
                    and inflight
                    and not self._cq.is_done
                ):
                    self._drain_frames(inflight)
                    self._retry_timeouts(inflight)
        finally:
            if not self._cq.is_done:
                for entry in list(inflight.values()):
                    self._cq.requeue(entry["chunk_idx"], entry["offset"], entry["length"])

        logger.info("Stream %d: worker loop exited", self.stream_id)

    def _fill_window(self, fh, inflight: dict[int, dict]) -> None:
        while (
            len(inflight) < PIPELINE_WINDOW
            and not self._stop_event.is_set()
            and not self._cq.is_done
        ):
            item = self._cq.get(timeout=0.0)
            if item is None:
                break

            chunk_idx, offset, length = item

            try:
                fh.seek(offset)
                data = fh.read(length)
                self._send_chunk(chunk_idx, offset, data)
            except (ConnectionError, OSError) as exc:
                logger.error("Stream %d: socket error: %s", self.stream_id, exc)
                self._stop_event.set()
                self._cq.requeue(chunk_idx, offset, length)
                return

            inflight[chunk_idx] = {
                "chunk_idx": chunk_idx,
                "offset": offset,
                "length": length,
                "data": data,
                "sent_at": time.monotonic(),
                "retries": 0,
            }

    def _send_chunk(self, chunk_idx: int, offset: int, data: bytes) -> None:
        frame = encode_frame(
            msg_type=MessageType.CHUNK,
            stream_id=self.stream_id,
            chunk_idx=chunk_idx,
            total_chunks=self._total_chunks,
            seq=chunk_idx,
            file_offset=offset,
            payload=data,
        )
        self._sock.sendall(frame)
        self._sm.record_bytes(len(data))
        self._cq.mark_sent()

    def _drain_frames(self, inflight: dict[int, dict]) -> None:
        for _ in range(PIPELINE_WINDOW * 2):
            try:
                hdr, _payload = recv_frame(self._sock)
            except socket.timeout:
                return
            except (ConnectionError, OSError) as exc:
                logger.error("Stream %d: socket error: %s", self.stream_id, exc)
                self._stop_event.set()
                return

            self._handle_frame(hdr, inflight)

    def _handle_frame(self, hdr: dict, inflight: dict[int, dict]) -> None:
        msg = hdr["msg_type"]
        if msg == MessageType.PONG:
            self._sm.record_pong()
            logger.debug("Stream %d: PONG received", self.stream_id)
            return

        if msg == MessageType.CHUNK_ACK:
            chunk_idx = hdr["chunk_idx"]
            entry = inflight.pop(chunk_idx, None)
            if entry is None:
                logger.debug(
                    "Stream %d: late/duplicate ACK for chunk %d",
                    self.stream_id, chunk_idx,
                )
                return

            self._cq.mark_acked()
            logger.debug(
                "Stream %d chunk %d ACKed in %.1f ms",
                self.stream_id,
                chunk_idx,
                (time.monotonic() - entry["sent_at"]) * 1000.0,
            )
            return

        logger.warning(
            "Stream %d: unexpected response %s",
            self.stream_id, msg,
        )

    def _retry_timeouts(self, inflight: dict[int, dict]) -> None:
        if not inflight:
            return

        now = time.monotonic()
        for chunk_idx, entry in list(inflight.items()):
            if now - entry["sent_at"] < ACK_TIMEOUT:
                continue

            if entry["retries"] + 1 >= MAX_CHUNK_RETRIES:
                logger.error(
                    "Stream %d chunk %d: failed after %d attempts, requeueing",
                    self.stream_id,
                    chunk_idx,
                    MAX_CHUNK_RETRIES,
                )
                self._sm.record_nack()
                inflight.pop(chunk_idx, None)
                if not self._cq.is_done:
                    self._cq.requeue(chunk_idx, entry["offset"], entry["length"])
                continue

            try:
                self._send_chunk(chunk_idx, entry["offset"], entry["data"])
            except (ConnectionError, OSError) as exc:
                logger.error("Stream %d: socket error during retry: %s", self.stream_id, exc)
                self._stop_event.set()
                return

            entry["retries"] += 1
            entry["sent_at"] = now
            self._sm.record_nack()
            logger.warning(
                "Stream %d chunk %d: ACK timeout, retry %d/%d",
                self.stream_id,
                chunk_idx,
                entry["retries"],
                MAX_CHUNK_RETRIES,
            )

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
            "progress_pct":  (
                round(acked / total * 100, 1)
                if total
                else (100.0 if self._chunk_queue.is_done else 0.0)
            ),
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
        last_progress_print = 0.0
        while not self._done_event.is_set():
            acked, total = self._chunk_queue.progress

            # Print progress bar
            pct = acked / total if total else (1.0 if self._chunk_queue.is_done else 0.0)
            bar_len = 40
            filled = int(bar_len * pct)
            bar = "█" * filled + "░" * (bar_len - filled)
            elapsed = time.monotonic() - self._start_time
            mbps = (acked * CHUNK_SIZE * 8) / (elapsed * 1e6) if elapsed > 0 else 0

            now = time.monotonic()
            if (
                now - last_progress_print >= PROGRESS_UPDATE_INTERVAL
                or self._chunk_queue.is_done
            ):
                print(
                    f"\r  [{bar}] {pct*100:5.1f}%  {acked}/{total} chunks  "
                    f"{mbps:.1f} Mbps  {len(self._workers)} streams  ",
                    end="",
                    flush=True,
                )
                last_progress_print = now

            if self._chunk_queue.wait_complete(timeout=1.0):
                self._end_time = time.monotonic()
                print()  # newline after progress bar
                duration = max(1e-9, self._end_time - self._start_time)
                logger.info(
                    "Transfer %s complete in %.2f s  (%.2f Mbps)",
                    self._transfer_id,
                    duration,
                    (self._file_size * 8) / (duration * 1e6),
                )

                # FIX (Bug 1): Send TRANSFER_DONE on every active stream and wait
                # for the server's COMPLETE response BEFORE closing the socket.
                # Previously, _stop_stream() closed the socket immediately after
                # all chunks were ACKed, while the server was still blocked on
                # recv_frame() expecting the next message. That abrupt close
                # caused WinError 10054 (connection reset) / 10053 (connection
                # aborted) on the server side, and the last chunk appeared to
                # not be transmitted because the server-side FileAssembler never
                # reached its is_complete state cleanly.
                with self._lock:
                    ids = list(self._workers.keys())

                for sid in ids:
                    with self._lock:
                        worker = self._workers.get(sid)
                    if worker:
                        worker.send_transfer_done()   # ← graceful protocol close
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

