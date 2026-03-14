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

ACK_TIMEOUT     = 60.0   # seconds — raised from 30 s; RTT on a loaded Wi-Fi
                          # LAN can spike to ~30 s so 30 s was too tight.
PING_INTERVAL   = 4.0    # seconds between RTT probes per stream
CONNECT_TIMEOUT = 20.0   # seconds
MAX_CHUNK_RETRIES = 3    # retransmit attempts before giving up

# BUG D FIX: pipeline_depth was logged but never used — the worker was
# already doing send-one / wait-ACK-one. This constant documents that.
PIPELINE_DEPTH  = 1      # send-one-wait-ACK (no pipelining)


class ChunkQueue:
    """Thread-safe queue of (chunk_idx, file_offset, length) tuples."""

    def __init__(self, file_path: str, chunk_size: int = CHUNK_SIZE) -> None:
        self.file_path    = file_path
        self.chunk_size   = chunk_size
        self.file_size    = os.path.getsize(file_path)
        self.total_chunks = math.ceil(self.file_size / chunk_size)

        self._q         = queue.Queue()
        self._completed = threading.Event()
        self._lock      = threading.Lock()
        self._sent_count  = 0
        self._acked_count = 0
        self._acked_chunks: set[int] = set()

        for idx in range(self.total_chunks):
            offset = idx * chunk_size
            length = min(chunk_size, self.file_size - offset)
            self._q.put((idx, offset, length))

        if self.total_chunks == 0:
            self._completed.set()

    def get(self, timeout: float = 1.0) -> Optional[tuple]:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                item = self._q.get(timeout=remaining)
            except queue.Empty:
                return None

            chunk_idx, offset, length = item
            with self._lock:
                if chunk_idx in self._acked_chunks:
                    continue
            return (chunk_idx, offset, length)

    def requeue(self, chunk_idx: int, offset: int, length: int) -> None:
        with self._lock:
            if chunk_idx in self._acked_chunks:
                return
        self._q.put((chunk_idx, offset, length))

    def mark_acked(self, chunk_idx: int) -> bool:
        with self._lock:
            if chunk_idx in self._acked_chunks:
                return False
            self._acked_chunks.add(chunk_idx)
            self._acked_count += 1
            if self._acked_count >= self.total_chunks:
                self._completed.set()
            return True

    def mark_sent(self) -> None:
        with self._lock:
            self._sent_count += 1

    def is_acked(self, chunk_idx: int) -> bool:
        with self._lock:
            return chunk_idx in self._acked_chunks

    def wait_complete(self, timeout: float = None) -> bool:
        return self._completed.wait(timeout=timeout)

    @property
    def is_done(self) -> bool:
        return self._completed.is_set()

    @property
    def progress(self) -> tuple[int, int]:
        with self._lock:
            return self._acked_count, self.total_chunks

    @property
    def remaining_chunks(self) -> int:
        with self._lock:
            return max(0, self.total_chunks - self._acked_count)


class StreamWorker:
    """One parallel TCP stream connecting sender to receiver."""

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
        self.stream_id     = stream_id
        self.host          = host
        self.port          = port
        self._cq           = chunk_queue
        self._sm           = stream_metrics
        self._transfer_id  = transfer_id
        self._file_name    = file_name
        self._total_chunks = total_chunks
        self._file_size    = file_size
        self._pipeline_depth = max(1, int(pipeline_depth))

        self._stop_event       = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None
        self._sock_lock        = threading.Lock()
        self._last_ping_time: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Connect to receiver and start the worker thread."""
        try:
            sock = socket.create_connection(
                (self.host, self.port), timeout=CONNECT_TIMEOUT
            )
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(ACK_TIMEOUT)
            self._sock = sock
        except OSError as exc:
            logger.error("Stream %d: connect failed: %s", self.stream_id, exc)
            return False

        if not self._send_hello():
            self._close_sock()
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"stream-{self.stream_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Stream %d: started (pipeline=%d) -> %s:%d",
            self.stream_id, self._pipeline_depth, self.host, self.port,
        )
        return True

    def send_transfer_done(self) -> bool:
        """Send TRANSFER_DONE and wait for COMPLETE before closing."""
        with self._sock_lock:
            sock = self._sock
        if sock is None:
            return False
        try:
            frame = encode_frame(
                msg_type=MessageType.TRANSFER_DONE,
                stream_id=self.stream_id & 0xFFFF,
            )
            sock.sendall(frame)
            while True:
                hdr, payload = recv_frame(sock)
                if hdr["msg_type"] == MessageType.PONG:
                    continue
                if hdr["msg_type"] == MessageType.COMPLETE:
                    logger.info("Stream %d: COMPLETE received", self.stream_id)
                    return True
                if hdr["msg_type"] == MessageType.ERROR:
                    details = decode_meta(payload) if payload else {}
                    logger.error(
                        "Stream %d: receiver reported incomplete transfer: %s",
                        self.stream_id,
                        details,
                    )
                    return False
                logger.warning(
                    "Stream %d: unexpected TRANSFER_DONE response: %s",
                    self.stream_id,
                    hdr["msg_type"],
                )
                return False
        except OSError as exc:
            logger.warning("Stream %d: send_transfer_done error: %s", self.stream_id, exc)
            return False

    def stop(self) -> None:
        """
        Signal this stream to stop and close its socket immediately.

        BUG C FIX: removed thread.join(timeout=32s). Closing the socket
        is enough — the worker thread wakes from its blocked recv/send
        and exits naturally. Joining was blocking the caller for up to
        32 s per stream (160 s when removing 5 streams at once).
        """
        self._stop_event.set()
        self._close_sock()
        logger.info("Stream %d: stop signalled", self.stream_id)

    @property
    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _close_sock(self) -> None:
        """Thread-safe socket close — safe to call from any thread."""
        with self._sock_lock:
            sock = self._sock
            self._sock = None
        if sock:
            try:
                sock.close()
            except OSError:
                pass

    def _get_live_sock(self) -> Optional[socket.socket]:
        with self._sock_lock:
            return self._sock

    def _send_chunk_frame(
        self,
        fh,
        chunk_idx: int,
        offset: int,
        length: int,
    ) -> float:
        """
        Send one chunk frame and return the send timestamp.
        Raises OSError/ConnectionError if the socket is no longer usable.
        """
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

        sock = self._get_live_sock()
        if sock is None:
            raise OSError("stream socket closed")

        sent_at = time.monotonic()
        sock.sendall(frame)
        self._sm.record_bytes(len(data))
        self._cq.mark_sent()
        return sent_at

    def _requeue_pending(self, pending: dict[int, dict]) -> None:
        for chunk_idx, entry in pending.items():
            if not self._cq.is_acked(chunk_idx):
                self._cq.requeue(chunk_idx, entry["offset"], entry["length"])

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
            stream_id=self.stream_id & 0xFFFF,
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
        Worker loop: keep up to pipeline_depth chunks in flight, then drain ACKs.

        BUG 1 FIX: entire body in try/except Exception so any unexpected
        error (e.g. file open failure on Windows with special characters in
        the path) is logged and the socket is closed cleanly.

        BUG A FIX: on ConnectionError/OSError inside the retry loop,
        _close_sock() is called immediately so the server's recv_frame()
        unblocks right away rather than waiting STREAM_TIMEOUT (120 s)
        and then sending WinError 10054 back to the sender.
        """
        pending: dict[int, dict] = {}

        try:
            with open(self._cq.file_path, "rb") as fh:
                while not self._stop_event.is_set() and not self._cq.is_done:
                    self._maybe_ping()

                    while (
                        len(pending) < self._pipeline_depth
                        and not self._stop_event.is_set()
                        and not self._cq.is_done
                    ):
                        item = self._cq.get(timeout=0.1 if pending else 0.5)
                        if item is None:
                            break

                        chunk_idx, offset, length = item

                        try:
                            sent_at = self._send_chunk_frame(fh, chunk_idx, offset, length)
                        except (ConnectionError, OSError) as exc:
                            if self._stop_event.is_set():
                                logger.debug(
                                    "Stream %d: socket closed during shutdown (expected): %s",
                                    self.stream_id, exc,
                                )
                            else:
                                logger.error("Stream %d: socket error during send: %s", self.stream_id, exc)
                            self._cq.requeue(chunk_idx, offset, length)
                            self._close_sock()
                            self._stop_event.set()
                            break

                        pending[chunk_idx] = {
                            "offset": offset,
                            "length": length,
                            "sent_at": sent_at,
                            "attempts": 1,
                        }

                    if not pending:
                        continue

                    sock = self._get_live_sock()
                    if sock is None:
                        self._stop_event.set()
                        break

                    try:
                        while True:
                            hdr, _payload = recv_frame(sock)
                            if hdr["msg_type"] == MessageType.PONG:
                                self._sm.record_pong()
                                continue
                            break

                        if hdr["msg_type"] == MessageType.CHUNK_ACK:
                            ack_idx = hdr["chunk_idx"]
                            entry = pending.pop(ack_idx, None)
                            if entry is None:
                                logger.debug(
                                    "Stream %d: ACK for unknown chunk %d ignored",
                                    self.stream_id, ack_idx,
                                )
                                continue

                            self._cq.mark_acked(ack_idx)
                            logger.debug(
                                "Stream %d chunk %d ACKed in %.1f ms",
                                self.stream_id, ack_idx,
                                (time.monotonic() - entry["sent_at"]) * 1000,
                            )
                            continue

                        logger.warning(
                            "Stream %d: unexpected response while %d chunks in flight: %s",
                            self.stream_id, len(pending), hdr["msg_type"],
                        )
                        self._sm.record_nack()

                    except socket.timeout:
                        oldest_idx, oldest = next(iter(pending.items()))
                        if oldest["attempts"] >= MAX_CHUNK_RETRIES:
                            logger.error(
                                "Stream %d chunk %d: ACK timeout after %d attempts",
                                self.stream_id, oldest_idx, oldest["attempts"],
                            )
                            self._sm.record_nack()
                            self._requeue_pending(pending)
                            pending.clear()
                            self._close_sock()
                            self._stop_event.set()
                            break

                        try:
                            resent_at = self._send_chunk_frame(
                                fh,
                                oldest_idx,
                                oldest["offset"],
                                oldest["length"],
                            )
                        except (ConnectionError, OSError) as exc:
                            logger.error(
                                "Stream %d chunk %d: resend failed: %s",
                                self.stream_id, oldest_idx, exc,
                            )
                            self._sm.record_nack()
                            self._requeue_pending(pending)
                            pending.clear()
                            self._close_sock()
                            self._stop_event.set()
                            break

                        oldest["attempts"] += 1
                        oldest["sent_at"] = resent_at
                        self._sm.record_nack()
                        logger.warning(
                            "Stream %d chunk %d: ACK timeout, resending (%d/%d)",
                            self.stream_id, oldest_idx, oldest["attempts"], MAX_CHUNK_RETRIES,
                        )

                    except (ConnectionError, OSError) as exc:
                        if self._stop_event.is_set():
                            logger.debug(
                                "Stream %d: socket closed during shutdown (expected): %s",
                                self.stream_id, exc,
                            )
                        else:
                            logger.error("Stream %d: socket error: %s", self.stream_id, exc)
                        self._requeue_pending(pending)
                        pending.clear()
                        self._close_sock()
                        self._stop_event.set()
                        break

        except Exception as exc:
            # BUG 1 FIX: catch anything unexpected and log it.
            logger.error(
                "Stream %d: fatal error in worker thread: %s",
                self.stream_id, exc, exc_info=True,
            )
            self._close_sock()
            self._stop_event.set()

        if pending and not self._cq.is_done:
            self._requeue_pending(pending)

        logger.info("Stream %d: worker loop exited", self.stream_id)

    # ------------------------------------------------------------------
    # Internal: RTT probe
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
        pipeline_depth: int = PIPELINE_DEPTH,
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
        self._pipeline_depth = max(1, int(pipeline_depth))

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

        # Avoid scaling beyond remaining work near the end of transfer.
        remaining_chunks = self._chunk_queue.remaining_chunks
        upper_bound = max(1, remaining_chunks)
        new_count = max(1, min(int(new_count), upper_bound))

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
            pipeline_depth=self._pipeline_depth,
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
            # BUG C FIX: worker.stop() no longer blocks on thread join,
            # so calling it directly here is safe and fast.
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
            bar     = "#" * filled + "-" * (bar_len - filled)
            elapsed = time.monotonic() - self._start_time
            acked_bytes = (self._file_size * acked / total) if total else 0
            mbps    = (acked_bytes * 8) / (elapsed * 1e6) if elapsed > 0 else 0

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
                        ok = worker.send_transfer_done()
                        if not ok and self._error is None:
                            self._error = "Receiver reported incomplete transfer"
                    self._stop_stream(sid)

                self.metrics.unregister_all()
                self._done_event.set()
                break

            with self._lock:
                alive = [w for w in self._workers.values() if w.is_alive]
            if not alive and not self._chunk_queue.is_done:
                logger.error("All stream workers died - transfer failed")
                self._error = "All stream workers died"
                self._done_event.set()
                break
