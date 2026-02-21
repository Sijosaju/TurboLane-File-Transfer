"""
turbolane_server/adapter.py

TurboLaneAdapter — the bridge between the file transfer server and the
TurboLane engine.

Responsibilities:
  1. Own the TurboLaneEngine instance (embedded mode).
  2. Run a background thread that fires every `interval` seconds.
  3. On each tick: collect metrics → call engine.decide() → call engine.learn()
     → invoke the stream_count_callback so the server can adjust streams.
  4. Expose get_stats() for the status subcommand.

Design contract:
  - Adapter never touches sockets, files, or transfer logic.
  - Server never touches the TurboLane engine directly.
  - The callback is the ONLY coupling point between the two layers.
"""

import threading
import time
import logging
from typing import Callable, Optional

from turbolane.engine import TurboLaneEngine
from turbolane_server.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class TurboLaneAdapter:
    """
    Embeds TurboLaneEngine inside the server process and drives the
    5-second monitoring / decision loop.

    Args:
        metrics:              MetricsCollector shared with the transfer layer
        stream_count_callback: Called with (new_stream_count: int) every cycle
        interval:             Seconds between TurboLane decisions (default 5)
        mode:                 TurboLane mode ('dci')
        algorithm:            RL algorithm ('qlearning')
        model_dir:            Where Q-table is persisted
        min_streams:          Minimum parallel TCP streams
        max_streams:          Maximum parallel TCP streams
        default_streams:      Starting stream count
    """

    def __init__(
        self,
        metrics: MetricsCollector,
        stream_count_callback: Callable[[int], None],
        interval: float = 5.0,
        mode: str = "dci",
        algorithm: str = "qlearning",
        model_dir: str = "models/dci",
        min_streams: int = 1,
        max_streams: int = 32,
        default_streams: int = 4,
    ) -> None:
        self._metrics   = metrics
        self._callback  = stream_count_callback
        self._interval  = interval

        # Instantiate the TurboLane engine (decoupled from all app logic)
        self._engine = TurboLaneEngine(
            mode=mode,
            algorithm=algorithm,
            model_dir=model_dir,
            min_connections=min_streams,
            max_connections=max_streams,
            default_connections=default_streams,
            monitoring_interval=interval,
        )

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Last snapshot for status display
        self._last_snapshot: dict = {}
        self._last_decision_streams: int = default_streams
        self._decision_count: int = 0

        logger.info(
            "TurboLaneAdapter ready: interval=%.1fs mode=%s algo=%s streams=[%d..%d]",
            interval, mode, algorithm, min_streams, max_streams,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background monitoring / decision thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("Adapter already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="turbolane-adapter",
            daemon=True,
        )
        self._thread.start()
        logger.info("TurboLaneAdapter started")

    def stop(self) -> None:
        """Signal the monitoring thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self._interval + 2)
        self._engine.save()
        logger.info("TurboLaneAdapter stopped; Q-table saved")

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """
        Main loop: sleep → collect → decide → learn → callback.

        The engine's internal `should_decide()` gate means that even if
        we call decide() faster than the interval, it only acts every
        `monitoring_interval` seconds. We sleep `interval` to be efficient.
        """
        # Give the transfer a moment to produce initial metrics
        time.sleep(self._interval)

        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as exc:
                logger.error("Adapter tick error: %s", exc, exc_info=True)

            # Sleep in small increments so stop_event is checked promptly
            deadline = time.monotonic() + self._interval
            while time.monotonic() < deadline and not self._stop_event.is_set():
                time.sleep(0.2)

    def _tick(self) -> None:
        """Single monitoring cycle."""
        snap = self._metrics.snapshot()
        self._last_snapshot = snap

        throughput = snap["throughput_mbps"]
        rtt        = snap["rtt_ms"]
        loss       = snap["loss_pct"]

        # --- Phase 1: learn from previous decision's outcome ---
        self._engine.learn(throughput, rtt, loss)

        # --- Phase 2: decide new stream count ---
        new_streams = self._engine.decide(throughput, rtt, loss)

        if new_streams != self._last_decision_streams:
            logger.info(
                "TurboLane: streams %d → %d  (tput=%.1f Mbps rtt=%.1f ms loss=%.3f%%)",
                self._last_decision_streams, new_streams,
                throughput, rtt, loss,
            )
            self._last_decision_streams = new_streams
            self._decision_count += 1

            # Notify the transfer layer
            try:
                self._callback(new_streams)
            except Exception as cb_exc:
                logger.error("stream_count_callback failed: %s", cb_exc)
        else:
            logger.debug(
                "TurboLane: streams hold at %d  (tput=%.1f rtt=%.1f loss=%.3f%%)",
                new_streams, throughput, rtt, loss,
            )

    # ------------------------------------------------------------------
    # Status / introspection
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """
        Return a combined stats dict for the `status` subcommand.
        Merges engine stats, last metrics snapshot, and adapter counters.
        """
        stats = self._engine.get_stats()
        stats.update({
            "adapter_interval_s":       self._interval,
            "adapter_decision_changes":  self._decision_count,
            "adapter_running":           bool(self._thread and self._thread.is_alive()),
            "last_throughput_mbps":      self._last_snapshot.get("throughput_mbps", 0.0),
            "last_rtt_ms":               self._last_snapshot.get("rtt_ms", 0.0),
            "last_loss_pct":             self._last_snapshot.get("loss_pct", 0.0),
            "last_active_streams":       self._last_snapshot.get("active_streams", 0),
        })
        return stats

    @property
    def current_streams(self) -> int:
        return self._engine.current_connections
