"""
turbolane_server/metrics.py

In-application network metrics collection.

RTT estimation:
    Each stream periodically sends a PING frame and records the time.
    When a PONG is received the RTT sample is recorded.
    No root / raw sockets required — all measured at the application layer.

Throughput:
    Tracked per-stream via bytes_sent and elapsed time.
    Aggregated across all active streams by MetricsCollector.

Packet loss (approximation):
    True packet loss is invisible to application code. We use chunk
    retransmit requests (NACK / timeout) as a proxy for loss events.
    The loss_pct exposed to TurboLane is:
        lost_chunks / total_chunks_sent * 100
"""

import time
import threading
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Number of RTT samples kept in the rolling window
RTT_WINDOW = 20


@dataclass
class StreamMetrics:
    """
    Per-stream counters maintained by each worker thread.
    All writes happen on the owning thread; reads happen on the
    MetricsCollector aggregator thread — protected by a lock.
    """
    stream_id: int
    lock: threading.Lock = field(default_factory=threading.Lock)

    # Throughput
    bytes_sent: int = 0
    start_time: float = field(default_factory=time.monotonic)
    last_snapshot_bytes: int = 0
    last_snapshot_time: float = field(default_factory=time.monotonic)

    # RTT
    rtt_samples: deque = field(default_factory=lambda: deque(maxlen=RTT_WINDOW))
    _ping_sent_at: Optional[float] = None

    # Loss proxy
    chunks_sent: int = 0
    chunks_nacked: int = 0       # retransmit requests received

    def record_bytes(self, n: int) -> None:
        with self.lock:
            self.bytes_sent += n
            self.chunks_sent += 1

    def record_nack(self) -> None:
        with self.lock:
            self.chunks_nacked += 1

    def record_ping_sent(self) -> None:
        with self.lock:
            self._ping_sent_at = time.monotonic()

    def record_pong(self) -> None:
        with self.lock:
            if self._ping_sent_at is not None:
                rtt_ms = (time.monotonic() - self._ping_sent_at) * 1000.0
                self.rtt_samples.append(rtt_ms)
                self._ping_sent_at = None
                logger.debug("Stream %d RTT sample: %.2f ms", self.stream_id, rtt_ms)

    def snapshot_throughput_mbps(self) -> float:
        """
        Instantaneous throughput since last snapshot (Mbps).
        Call periodically (e.g., every 5 s) from the aggregator.
        """
        with self.lock:
            now = time.monotonic()
            dt = now - self.last_snapshot_time
            if dt <= 0:
                return 0.0
            delta_bytes = self.bytes_sent - self.last_snapshot_bytes
            self.last_snapshot_bytes = self.bytes_sent
            self.last_snapshot_time = now
            return (delta_bytes * 8) / (dt * 1e6)   # bits → Mbps

    def mean_rtt_ms(self) -> float:
        with self.lock:
            if not self.rtt_samples:
                return 0.0
            return sum(self.rtt_samples) / len(self.rtt_samples)

    def loss_pct(self) -> float:
        with self.lock:
            if self.chunks_sent == 0:
                return 0.0
            return (self.chunks_nacked / self.chunks_sent) * 100.0


class MetricsCollector:
    """
    Aggregates metrics across all active streams and exposes the
    three values TurboLane needs:

        throughput_mbps  — sum of per-stream instantaneous throughput
        rtt_ms           — average mean RTT across streams (ms)
        loss_pct         — weighted loss proxy across streams

    Thread-safe: streams register/unregister dynamically.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._streams: dict[int, StreamMetrics] = {}

        # History for display / logging
        self._history: deque = deque(maxlen=200)

    # ------------------------------------------------------------------
    # Stream lifecycle
    # ------------------------------------------------------------------

    def register_stream(self, stream_id: int) -> StreamMetrics:
        sm = StreamMetrics(stream_id=stream_id)
        with self._lock:
            self._streams[stream_id] = sm
        logger.debug("MetricsCollector: registered stream %d", stream_id)
        return sm

    def unregister_stream(self, stream_id: int) -> None:
        with self._lock:
            self._streams.pop(stream_id, None)
        logger.debug("MetricsCollector: unregistered stream %d", stream_id)

    def unregister_all(self) -> None:
        with self._lock:
            self._streams.clear()

    # ------------------------------------------------------------------
    # Aggregated snapshot (called every 5 s by the TurboLane adapter)
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """
        Return aggregated metrics snapshot.

        Returns:
            dict with keys: throughput_mbps, rtt_ms, loss_pct,
                            active_streams, timestamp
        """
        with self._lock:
            streams = list(self._streams.values())

        if not streams:
            return {
                "throughput_mbps": 0.0,
                "rtt_ms":          0.0,
                "loss_pct":        0.0,
                "active_streams":  0,
                "timestamp":       time.monotonic(),
            }

        total_throughput = sum(s.snapshot_throughput_mbps() for s in streams)

        rtt_values = [s.mean_rtt_ms() for s in streams if s.mean_rtt_ms() > 0]
        avg_rtt = sum(rtt_values) / len(rtt_values) if rtt_values else 0.0

        total_sent  = sum(s.chunks_sent  for s in streams)
        total_nacked = sum(s.chunks_nacked for s in streams)
        loss = (total_nacked / total_sent * 100.0) if total_sent > 0 else 0.0

        result = {
            "throughput_mbps": round(total_throughput, 3),
            "rtt_ms":          round(avg_rtt, 3),
            "loss_pct":        round(loss, 4),
            "active_streams":  len(streams),
            "timestamp":       time.monotonic(),
        }

        self._history.append(result)
        logger.debug("Metrics snapshot: %s", result)
        return result

    def get_history(self) -> list:
        return list(self._history)
