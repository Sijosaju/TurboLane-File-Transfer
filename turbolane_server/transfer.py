"""
turbolane_server/transfer.py

TransferSession — orchestrates a single file transfer on the sender side.

FIXES IN THIS VERSION:
  BUG 1 — _run(): entire body wrapped in try/except so file-open failures
           are logged instead of silently killing the thread.
  BUG 2 — adjust_streams(): calls _prune_dead_workers() first so the live
           stream count is accurate and the adapter can't flood new streams.
  BUG 3 — _monitor_loop(): calls _prune_dead_workers() every iteration.
  BUG 4 — start(): aborts immediately if zero initial streams connected.
  BUG 5 — _run() retry loop: honours stop_event at the top of each attempt.
  BUG A — _run() socket error handler: closes self._sock immediately when a
           ConnectionError/OSError kills a worker, so the server-side
           StreamHandler unblocks right away instead of waiting 120 s
           (STREAM_TIMEOUT) before issuing WinError 10054 on the sender.
  BUG C — stop(): no longer joins the worker thread (join timeout was
           ACK_TIMEOUT + 2 = 32 s; removing 5 streams in one adapter tick
           used to freeze the transfer for up to 160 s). The socket is
           closed immediately so the thread wakes up from its next
           recv/send and exits on its own.

PIPELINE REFACTOR (replaces stop-and-wait with a sliding window):
  Previously each StreamWorker sent one chunk, blocked waiting for its ACK,
  then sent the next. On a 20 ms RTT link that caps one stream at:
      512 KB / 0.020 s ≈ 200 Mbps theoretical, much less in practice
  because the socket was idle during every round-trip wait.

  Now each StreamWorker maintains a sliding window of PIPELINE_DEPTH chunks
  that are all in-flight simultaneously on the same TCP connection:

    ┌─────────────────────────────────────────────────────┐
    │  in_flight window  (up to PIPELINE_DEPTH entries)   │
    │  chunk_idx → (offset, length, send_time, attempt)   │
    └─────────────────────────────────────────────────────┘
      • A dedicated sender thread keeps the window full by pulling
        from ChunkQueue and writing frames to the socket.
      • A dedicated receiver thread reads ACKs / PONGs from the socket
        and removes entries from the window, allowing the sender thread
        to push more chunks.
      • When an ACK does not arrive within CHUNK_ACK_TIMEOUT seconds the
        chunk is requeued (up to MAX_CHUNK_RETRIES times) so another
        stream (or the same stream after a brief pause) can retry it.
      • A shared threading.Event (_fatal_event) lets either thread signal
        the other to exit cleanly on socket error.

  Net effect: the TCP pipe is kept full continuously instead of draining
  to zero after every chunk.  Expected gain on a 20 ms LAN with
  PIPELINE_DEPTH = 8: ~6–8× throughput improvement.
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
    HEADER_SIZE,
    encode_frame,
    encode_meta,
    decode_meta,
    recv_frame,
)
from turbolane_server.metrics import MetricsCollector, StreamMetrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

# How many chunks can be simultaneously in-flight on a single stream.
# Each chunk is CHUNK_SIZE bytes (512 KB), so PIPELINE_DEPTH=8 means up to
# 4 MB of unacknowledged data per stream — well within a 4 MB SO_SNDBUF.
# Raise this if RTT is high; lower it under heavy packet loss.
PIPELINE_DEPTH = 8

# Per-chunk ACK deadline inside the pipeline.  Much tighter than the old
# monolithic ACK_TIMEOUT because we now have per-chunk send timestamps.
# Set to 3× the worst expected RTT your tc-netem experiment will impose.
CHUNK_ACK_TIMEOUT = 5.0     # seconds

# Legacy socket-level timeout (used only for the HELLO handshake and
# send_transfer_done, where we still do a synchronous recv).
ACK_TIMEOUT     = 60.0

PING_INTERVAL   = 4.0       # seconds between RTT probes per stream
CONNECT_TIMEOUT = 20.0      # seconds
MAX_CHUNK_RETRIES = 3       # retransmit attempts before giving up on a chunk


# ---------------------------------------------------------------------------
# ChunkQueue  (unchanged from original)
# ---------------------------------------------------------------------------

class ChunkQueue:
    """Thread-safe queue of (chunk_idx, file_offset, length) tuples."""

    def __init__(self, file_path: str, chunk_size: int = CHUNK_SIZE) -> None:
        self.file_path    = file_path
        self.chunk_size   = chunk_size
        self.file_size    = os.path.getsize(file_path)
        self.total_chunks = math.ceil(self.file_size / chunk_size)

        self._q           = queue.Queue()
        self._completed   = threading.Event()
        self._lock        = threading.Lock()
        self._sent_count  = 0
        self._acked_count = 0

        for idx in range(self.total_chunks):
            offset = idx * chunk_size
            length = min(chunk_size, self.file_size - offset)
            self._q.put((idx, offset, length))

        if self.total_chunks == 0:
            self._completed.set()

    def get(self, timeout: float = 1.0) -> Optional[tuple]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def requeue(self, chunk_idx: int, offset: int, length: int) -> None:
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


# ---------------------------------------------------------------------------
# StreamWorker  (pipelined sliding-window redesign)
# ---------------------------------------------------------------------------

class StreamWorker:
    """
    One parallel TCP stream connecting sender to receiver.

    Internal threading model
    ────────────────────────
    Two daemon threads share a single TCP socket:

      _sender_thread  — pulls chunks from ChunkQueue, writes frames to socket,
                        records send timestamps in _in_flight.
      _receiver_thread— reads ACKs/PONGs from socket, removes entries from
                        _in_flight, calls mark_acked on the ChunkQueue.

    They communicate through:
      _in_flight      : dict[chunk_idx → _InFlightEntry]  (protected by _ifl_lock)
      _window_sem     : Semaphore(PIPELINE_DEPTH) — sender blocks here when window full
      _fatal_event    : threading.Event — set by either thread on fatal socket error

    PING frames are injected by the sender thread and PONG frames are consumed
    by the receiver thread — same as before, no protocol change.
    """

    # Small named-tuple-style container stored in _in_flight
    class _InFlightEntry:
        __slots__ = ("offset", "length", "send_time", "attempt")
        def __init__(self, offset: int, length: int, send_time: float, attempt: int):
            self.offset    = offset
            self.length    = length
            self.send_time = send_time
            self.attempt   = attempt

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
        pipeline_depth: int = PIPELINE_DEPTH,
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
        self._pipeline_depth = pipeline_depth

        self._stop_event    = threading.Event()
        self._fatal_event   = threading.Event()   # socket-level fatal error

        self._sock: Optional[socket.socket] = None
        self._sock_lock     = threading.Lock()

        # Sliding window state
        # chunk_idx → _InFlightEntry
        self._in_flight: dict[int, "StreamWorker._InFlightEntry"] = {}
        self._ifl_lock  = threading.Lock()

        # Semaphore limits how many chunks the sender thread can have in flight
        self._window_sem = threading.Semaphore(pipeline_depth)

        # Retry counters per chunk: chunk_idx → attempt_number
        self._retry_counts: dict[int, int] = {}
        self._retry_lock   = threading.Lock()

        self._sender_thread:   Optional[threading.Thread] = None
        self._receiver_thread: Optional[threading.Thread] = None
        self._last_ping_time: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Connect to receiver and start sender + receiver threads."""
        try:
            sock = socket.create_connection(
                (self.host, self.port), timeout=CONNECT_TIMEOUT
            )
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # LAN optimization: 4 MB kernel buffers per stream
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
            # No monolithic socket timeout — the pipeline manages its own
            # per-chunk deadlines via CHUNK_ACK_TIMEOUT.  We use a short
            # recv timeout only so the receiver thread can wake up and check
            # _stop_event / _fatal_event periodically.
            sock.settimeout(1.0)
            self._sock = sock
        except OSError as exc:
            logger.error("Stream %d: connect failed: %s", self.stream_id, exc)
            return False

        if not self._send_hello():
            self._close_sock()
            return False

        self._stop_event.clear()
        self._fatal_event.clear()

        self._receiver_thread = threading.Thread(
            target=self._receiver_loop,
            name=f"stream-{self.stream_id}-rx",
            daemon=True,
        )
        self._sender_thread = threading.Thread(
            target=self._sender_loop,
            name=f"stream-{self.stream_id}-tx",
            daemon=True,
        )

        # Start receiver first so ACKs are consumed before sender fills pipe
        self._receiver_thread.start()
        self._sender_thread.start()

        logger.info(
            "Stream %d: started (pipeline_depth=%d) -> %s:%d",
            self.stream_id, self._pipeline_depth, self.host, self.port,
        )
        return True

    def send_transfer_done(self) -> None:
        """Send TRANSFER_DONE and wait for COMPLETE (called after all chunks ACKed)."""
        with self._sock_lock:
            sock = self._sock
        if sock is None:
            return
        try:
            # Restore a longer timeout for this synchronous handshake
            sock.settimeout(ACK_TIMEOUT)
            frame = encode_frame(
                msg_type=MessageType.TRANSFER_DONE,
                stream_id=self.stream_id & 0xFFFF,
            )
            sock.sendall(frame)
            while True:
                hdr, _payload = recv_frame(sock)
                if hdr["msg_type"] == MessageType.PONG:
                    continue
                if hdr["msg_type"] == MessageType.COMPLETE:
                    logger.info("Stream %d: COMPLETE received", self.stream_id)
                break
        except OSError as exc:
            logger.warning("Stream %d: send_transfer_done error: %s", self.stream_id, exc)

    def stop(self) -> None:
        """
        Signal both threads to stop and close the socket immediately.

        BUG C preserved: we do NOT join the threads.  Closing the socket
        makes both threads unblock from their next recv/send and exit.
        """
        self._stop_event.set()
        self._fatal_event.set()   # unblocks sender if it's waiting on _window_sem
        # Release semaphore enough times to unblock a waiting sender thread
        for _ in range(self._pipeline_depth):
            try:
                self._window_sem.release()
            except ValueError:
                break
        self._close_sock()
        logger.info("Stream %d: stop signalled", self.stream_id)

    @property
    def is_alive(self) -> bool:
        """True if at least one of the two worker threads is still running."""
        s_alive = bool(self._sender_thread   and self._sender_thread.is_alive())
        r_alive = bool(self._receiver_thread and self._receiver_thread.is_alive())
        return s_alive or r_alive

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _close_sock(self) -> None:
        with self._sock_lock:
            sock = self._sock
            self._sock = None
        if sock:
            try:
                sock.close()
            except OSError:
                pass

    def _fatal(self) -> None:
        """Mark stream as fatally failed; both threads will see this and exit."""
        self._fatal_event.set()
        self._stop_event.set()
        self._close_sock()
        # Drain the semaphore blockage so the sender thread can exit
        for _ in range(self._pipeline_depth):
            try:
                self._window_sem.release()
            except ValueError:
                break

    # ------------------------------------------------------------------
    # HELLO handshake  (synchronous, before threads start)
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
            stream_id=self.stream_id & 0xFFFF,
            payload=encode_meta(meta),
        )
        try:
            self._sock.settimeout(CONNECT_TIMEOUT)
            self._sock.sendall(frame)
            hdr, payload = recv_frame(self._sock)
            if hdr["msg_type"] == MessageType.BUSY:
                info = decode_meta(payload) if payload else {}
                logger.error(
                    "Stream %d: server BUSY — %s", self.stream_id, info.get("reason", "")
                )
                return False
            if hdr["msg_type"] != MessageType.HELLO_ACK:
                logger.error(
                    "Stream %d: unexpected HELLO response: %s",
                    self.stream_id, hdr["msg_type"],
                )
                return False
            return True
        except Exception as exc:
            logger.error("Stream %d: HELLO failed: %s", self.stream_id, exc)
            return False

    # ------------------------------------------------------------------
    # Sender thread
    # ------------------------------------------------------------------

    def _sender_loop(self) -> None:
        """
        Pull chunks from ChunkQueue and push them onto the wire, keeping
        up to PIPELINE_DEPTH chunks in-flight simultaneously.

        The semaphore _window_sem starts at PIPELINE_DEPTH.  Before sending
        each chunk the sender acquires one slot; the receiver thread releases
        a slot for each ACK it processes.  This gives us a clean credit-based
        flow-control mechanism with no busy-polling.
        """
        try:
            with open(self._cq.file_path, "rb") as fh:
                while not self._stop_event.is_set() and not self._fatal_event.is_set():

                    if self._cq.is_done:
                        break

                    # ── Periodic timeout sweep ──────────────────────────────
                    # Check whether any in-flight chunk has exceeded its
                    # per-chunk ACK deadline.  If so, requeue it so another
                    # stream (or this one after the fatal-window-refill) can
                    # retry it.  We do this before blocking on the semaphore.
                    self._sweep_timeouts(fh)

                    # ── Wait for a free window slot ─────────────────────────
                    # Use a timed acquire so we can loop back and run the
                    # timeout sweep even when the window is full.
                    acquired = self._window_sem.acquire(timeout=0.5)
                    if not acquired:
                        # Window full — loop back, run sweep, try again
                        continue

                    if self._stop_event.is_set() or self._fatal_event.is_set():
                        # Release slot we just acquired; we're exiting
                        self._window_sem.release()
                        break

                    # ── Pull next chunk ─────────────────────────────────────
                    item = self._cq.get(timeout=0.3)
                    if item is None:
                        # Queue temporarily empty (all chunks either in-flight
                        # or already ACKed).  Release the slot and wait.
                        self._window_sem.release()
                        continue

                    chunk_idx, offset, length = item

                    # ── Get retry count for this chunk ──────────────────────
                    with self._retry_lock:
                        attempt = self._retry_counts.get(chunk_idx, 0)

                    if attempt >= MAX_CHUNK_RETRIES:
                        logger.error(
                            "Stream %d chunk %d: exceeded max retries (%d), dropping",
                            self.stream_id, chunk_idx, MAX_CHUNK_RETRIES,
                        )
                        # Release window slot — we're not putting this in-flight
                        self._window_sem.release()
                        # Still mark it acked so the transfer can complete
                        # (the receiver side won't see an ACK for this chunk).
                        # Alternatively surface as an error — for now we log and skip.
                        continue

                    # ── Inject PING if due ──────────────────────────────────
                    self._maybe_ping()

                    # ── Send the chunk ──────────────────────────────────────
                    with self._sock_lock:
                        sock = self._sock
                    if sock is None:
                        self._window_sem.release()
                        self._cq.requeue(chunk_idx, offset, length)
                        break

                    try:
                        fh.seek(offset)
                        data = fh.read(length)

                        frame = encode_frame(
                            msg_type=MessageType.CHUNK,
                            stream_id=self.stream_id & 0xFFFF,
                            chunk_idx=chunk_idx,
                            total_chunks=self._total_chunks,
                            seq=chunk_idx,
                            file_offset=offset,
                            payload=data,
                        )

                        send_time = time.monotonic()
                        sock.sendall(frame)

                        # Only record bytes / sent on the first attempt
                        if attempt == 0:
                            self._sm.record_bytes(len(data))
                            self._cq.mark_sent()

                        # Register in the in-flight window
                        with self._ifl_lock:
                            self._in_flight[chunk_idx] = self._InFlightEntry(
                                offset=offset,
                                length=length,
                                send_time=send_time,
                                attempt=attempt,
                            )

                        logger.debug(
                            "Stream %d → chunk %d sent (attempt %d, in_flight=%d)",
                            self.stream_id, chunk_idx, attempt,
                            len(self._in_flight),
                        )

                    except (ConnectionError, OSError) as exc:
                        logger.error(
                            "Stream %d: socket error sending chunk %d: %s",
                            self.stream_id, chunk_idx, exc,
                        )
                        self._window_sem.release()
                        self._cq.requeue(chunk_idx, offset, length)
                        self._fatal()
                        break

        except Exception as exc:
            logger.error(
                "Stream %d: fatal error in sender thread: %s",
                self.stream_id, exc, exc_info=True,
            )
            self._fatal()

        # ── Drain: requeue anything still in-flight when we exit ───────────
        self._requeue_all_in_flight()
        logger.info("Stream %d: sender thread exited", self.stream_id)

    # ------------------------------------------------------------------
    # Receiver thread
    # ------------------------------------------------------------------

    def _receiver_loop(self) -> None:
        """
        Read frames from the socket indefinitely.

        For each frame:
          CHUNK_ACK  → remove from _in_flight, release window slot, mark acked
          PONG       → record RTT sample
          anything else → log and ignore
        """
        while not self._stop_event.is_set() and not self._fatal_event.is_set():
            with self._sock_lock:
                sock = self._sock
            if sock is None:
                break

            try:
                hdr, _payload = recv_frame(sock)

            except socket.timeout:
                # Short recv timeout (1 s) — just loop and check stop events
                continue

            except (ConnectionError, OSError) as exc:
                if not self._stop_event.is_set():
                    logger.error(
                        "Stream %d: socket error in receiver thread: %s",
                        self.stream_id, exc,
                    )
                    self._fatal()
                break

            except Exception as exc:
                logger.error(
                    "Stream %d: unexpected error in receiver thread: %s",
                    self.stream_id, exc, exc_info=True,
                )
                self._fatal()
                break

            msg = hdr["msg_type"]

            if msg == MessageType.CHUNK_ACK:
                acked_idx = hdr["chunk_idx"]

                with self._ifl_lock:
                    entry = self._in_flight.pop(acked_idx, None)

                if entry is not None:
                    rtt_ms = (time.monotonic() - entry.send_time) * 1000
                    logger.debug(
                        "Stream %d ← ACK chunk %d  RTT %.1f ms  in_flight=%d",
                        self.stream_id, acked_idx, rtt_ms, len(self._in_flight),
                    )
                    # Clear retry counter on success
                    with self._retry_lock:
                        self._retry_counts.pop(acked_idx, None)

                    self._cq.mark_acked()
                    # Release one window slot so sender can push another chunk
                    self._window_sem.release()
                else:
                    # Stale / duplicate ACK — still safe to release a slot to
                    # avoid deadlock in case the sender was waiting
                    logger.debug(
                        "Stream %d: stale/duplicate ACK for chunk %d — discarding",
                        self.stream_id, acked_idx,
                    )
                    # Do NOT release the semaphore here — the slot was never
                    # consumed for this chunk (it was already acked and removed).
                    # Releasing would inflate the window beyond PIPELINE_DEPTH.

            elif msg == MessageType.PONG:
                self._sm.record_pong()

            else:
                logger.debug(
                    "Stream %d: ignoring unexpected frame type %s",
                    self.stream_id, msg,
                )

        # ── Drain: requeue anything still in-flight when receiver exits ─────
        self._requeue_all_in_flight()
        logger.info("Stream %d: receiver thread exited", self.stream_id)

    # ------------------------------------------------------------------
    # Timeout sweep  (called by sender thread)
    # ------------------------------------------------------------------

    def _sweep_timeouts(self, fh) -> None:
        """
        Walk _in_flight and requeue any chunk whose ACK deadline has passed.

        Called by the sender thread on every iteration of its loop, so it
        runs frequently without needing a dedicated timer thread.
        """
        now = time.monotonic()
        timed_out = []

        with self._ifl_lock:
            for idx, entry in list(self._in_flight.items()):
                if now - entry.send_time > CHUNK_ACK_TIMEOUT:
                    timed_out.append((idx, entry))

            for idx, _ in timed_out:
                del self._in_flight[idx]

        for idx, entry in timed_out:
            with self._retry_lock:
                attempt = self._retry_counts.get(idx, 0) + 1
                self._retry_counts[idx] = attempt

            logger.warning(
                "Stream %d chunk %d: ACK timeout (attempt %d/%d) — requeueing",
                self.stream_id, idx, attempt, MAX_CHUNK_RETRIES,
            )
            self._sm.record_nack()

            # Release the window slot that was held by this timed-out chunk
            self._window_sem.release()

            if attempt < MAX_CHUNK_RETRIES:
                self._cq.requeue(idx, entry.offset, entry.length)
            else:
                logger.error(
                    "Stream %d chunk %d: max retries reached — chunk lost",
                    self.stream_id, idx,
                )

    # ------------------------------------------------------------------
    # Drain helper
    # ------------------------------------------------------------------

    def _requeue_all_in_flight(self) -> None:
        """
        On stream exit, requeue all still-pending chunks so another stream
        (or a future stream) can deliver them.  Without this, in-flight chunks
        at exit time are silently lost causing the transfer to stall near 100%.
        """
        with self._ifl_lock:
            remaining = list(self._in_flight.items())
            self._in_flight.clear()

        for idx, entry in remaining:
            logger.debug(
                "Stream %d: requeueing in-flight chunk %d on exit",
                self.stream_id, idx,
            )
            self._cq.requeue(idx, entry.offset, entry.length)
            # Release slot so the semaphore count stays consistent
            self._window_sem.release()

    # ------------------------------------------------------------------
    # RTT probe  (injected by sender thread)
    # ------------------------------------------------------------------

    def _maybe_ping(self) -> None:
        now = time.monotonic()
        if now - self._last_ping_time < PING_INTERVAL:
            return
        self._last_ping_time = now
        with self._sock_lock:
            sock = self._sock
        if sock is None:
            return
        try:
            frame = encode_frame(
                msg_type=MessageType.PING,
                stream_id=self.stream_id & 0xFFFF,
            )
            self._sm.record_ping_sent()
            sock.sendall(frame)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# TransferSession  (unchanged public API; only _monitor_loop throughput fix)
# ---------------------------------------------------------------------------

class TransferSession:
    """Manages the complete lifecycle of one outbound file transfer."""

    def __init__(
        self,
        file_path: str,
        receiver_host: str,
        receiver_port: int,
        num_streams: int,
        metrics_collector: MetricsCollector,
        transfer_id: Optional[str] = None,
    ) -> None:
        self.file_path     = file_path
        self.receiver_host = receiver_host
        self.receiver_port = receiver_port
        self.metrics       = metrics_collector

        self._file_name   = Path(file_path).name
        self._file_size   = os.path.getsize(file_path)
        self._transfer_id = transfer_id or f"xfer-{int(time.time())}"

        self._chunk_queue = ChunkQueue(file_path)
        self._workers: dict[int, StreamWorker] = {}
        self._lock        = threading.Lock()
        self._target_streams = num_streams
        self._start_time: Optional[float] = None
        self._end_time:   Optional[float] = None
        self._done_event  = threading.Event()
        self._error: Optional[str] = None
        self._next_stream_id = 0

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
        """
        Spawn initial worker streams and begin transfer.

        BUG 4 FIX: abort immediately if zero streams connect.
        """
        self._start_time = time.monotonic()
        started = 0
        for _ in range(self._target_streams):
            if self._spawn_stream():
                started += 1

        if self._target_streams > 0 and started == 0:
            self._error = "All initial streams failed to connect"
            self._done_event.set()
            logger.error("TransferSession: no streams connected — aborting")
            return

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="transfer-monitor",
            daemon=True,
        )
        self._monitor_thread.start()

    def wait(self, timeout: float = None) -> bool:
        completed = self._done_event.wait(timeout=timeout)
        return completed and self._error is None

    def adjust_streams(self, new_count: int) -> None:
        """
        Called by the adapter when the RL engine recommends a stream count change.

        BUG 2 FIX: prune dead workers before counting so len(self._workers)
        only reflects live streams, preventing endless stream spawning.
        """
        if self._done_event.is_set() or self._chunk_queue.is_done:
            return

        # BUG 2 FIX: prune before counting.
        self._prune_dead_workers()

        with self._lock:
            current = len(self._workers)
            self._target_streams = new_count

        if new_count > current:
            for _ in range(new_count - current):
                logger.info("Adapter: adding stream (total -> %d)", new_count)
                self._spawn_stream()

        elif new_count < current:
            with self._lock:
                ids_to_stop = sorted(self._workers.keys(), reverse=True)
                ids_to_stop = ids_to_stop[:current - new_count]

            for sid in ids_to_stop:
                logger.info("Adapter: removing stream %d (total -> %d)", sid, new_count)
                self._stop_stream(sid)

    def get_stats(self) -> dict:
        elapsed = (
            (self._end_time or time.monotonic()) - self._start_time
            if self._start_time else 0.0
        )
        acked, total = self._chunk_queue.progress
        # Use actual file size for throughput — not acked*CHUNK_SIZE which
        # overcounts because the final chunk is smaller than CHUNK_SIZE.
        throughput = (self._file_size * 8) / (elapsed * 1e6) if elapsed > 0 else 0.0
        return {
            "transfer_id":     self._transfer_id,
            "file_name":       self._file_name,
            "file_size":       self._file_size,
            "total_chunks":    total,
            "acked_chunks":    acked,
            "progress_pct":    round(acked / total * 100, 1) if total else 0.0,
            "elapsed_s":       round(elapsed, 2),
            "throughput_mbps": round(throughput, 2),
            "active_streams":  self._active_worker_count(),
            "completed":       self._done_event.is_set(),
            "error":           self._error,
        }

    def abort(self) -> None:
        with self._lock:
            ids = list(self._workers.keys())
        for sid in ids:
            self._stop_stream(sid)
        self._error = "aborted"
        self._done_event.set()

    # ------------------------------------------------------------------
    # Internal: stream management
    # ------------------------------------------------------------------

    def _spawn_stream(self) -> bool:
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
            return True
        else:
            self.metrics.unregister_stream(stream_id)
            return False

    def _active_worker_count(self) -> int:
        with self._lock:
            return sum(1 for w in self._workers.values() if w.is_alive)

    def _prune_dead_workers(self) -> None:
        """Remove workers whose threads have exited from the workers dict."""
        with self._lock:
            dead_ids = [sid for sid, w in self._workers.items() if not w.is_alive]
        for sid in dead_ids:
            logger.warning("Pruning dead stream %d", sid)
            self._stop_stream(sid)

    def _stop_stream(self, stream_id: int) -> None:
        with self._lock:
            worker = self._workers.pop(stream_id, None)
        if worker:
            # BUG C FIX preserved: worker.stop() closes the socket immediately
            # and does NOT join threads, so this returns in microseconds.
            worker.stop()
            self.metrics.unregister_stream(stream_id)

    # ------------------------------------------------------------------
    # Internal: progress monitor
    # ------------------------------------------------------------------

    def _monitor_loop(self) -> None:
        """
        Watches transfer progress and signals completion.

        BUG 3 FIX: prune dead workers every iteration so the dict stays
        accurate and the "no alive workers" check below is reliable.
        """
        while not self._done_event.is_set():

            # BUG 3 FIX: prune every cycle.
            self._prune_dead_workers()

            acked, total = self._chunk_queue.progress

            pct     = acked / total if total else 0
            bar_len = 40
            filled  = int(bar_len * pct)
            bar     = "█" * filled + "░" * (bar_len - filled)
            elapsed = time.monotonic() - self._start_time

            # Use actual bytes transferred (not acked * CHUNK_SIZE) so the
            # last partial chunk doesn't inflate the displayed throughput.
            bytes_transferred = min(acked * self._chunk_queue.chunk_size, self._file_size)
            mbps = (bytes_transferred * 8) / (elapsed * 1e6) if elapsed > 0 else 0

            print(
                f"\r  [{bar}] {pct*100:5.1f}%  {acked}/{total} chunks  "
                f"{mbps:.1f} Mbps  {self._active_worker_count()} streams  ",
                end="", flush=True,
            )

            if self._chunk_queue.wait_complete(timeout=1.0):
                self._end_time = time.monotonic()
                print()
                logger.info(
                    "Transfer %s complete in %.2f s  (%.2f Mbps)",
                    self._transfer_id,
                    self._end_time - self._start_time,
                    (self._file_size * 8) / ((self._end_time - self._start_time) * 1e6),
                )

                with self._lock:
                    ids = list(self._workers.keys())

                for sid in ids:
                    with self._lock:
                        worker = self._workers.get(sid)
                    if worker:
                        worker.send_transfer_done()
                    self._stop_stream(sid)

                self.metrics.unregister_all()
                self._done_event.set()
                break

            with self._lock:
                alive = [w for w in self._workers.values() if w.is_alive]
            if not alive and not self._chunk_queue.is_done:
                logger.error("All stream workers died — transfer failed")
                self._error = "All stream workers died"
                self._done_event.set()
                break