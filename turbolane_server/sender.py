"""
turbolane_server/sender.py

TurboLane file transfer SENDER.

This is the active side:
  1. Validates the file and receiver
  2. Creates a MetricsCollector
  3. Creates a TransferSession (manages streams and chunk distribution)
  4. Creates a TurboLaneAdapter (embeds TurboLane, drives 5-second loop)
  5. Wires the adapter's callback to session.adjust_streams()
  6. Starts adapter + session, then waits for completion

No socket code here — that lives in transfer.py (StreamWorker).
No RL code here — that lives in adapter.py (TurboLaneAdapter).
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional

from turbolane_server.metrics import MetricsCollector
from turbolane_server.transfer import TransferSession
from turbolane_server.adapter import TurboLaneAdapter

logger = logging.getLogger(__name__)


class TurboLaneSender:
    """
    Orchestrates an outbound file transfer with embedded TurboLane optimization.

    Args:
        file_path:       Absolute or relative path to the file to send
        receiver_host:   Hostname / IP of the TurboLane server (receiver)
        receiver_port:   Port the server is listening on
        initial_streams: Starting number of parallel TCP streams
        min_streams:     Minimum streams (passed to TurboLane)
        max_streams:     Maximum streams (passed to TurboLane)
        model_dir:       Directory for Q-table persistence
        monitor_interval: Seconds between TurboLane decisions
        timeout:         Max seconds to wait for transfer completion (None = unlimited)
    """

    def __init__(
        self,
        file_path: str,
        receiver_host: str,
        receiver_port: int = 9000,
        initial_streams: int = 4,
        min_streams: int = 1,
        max_streams: int = 32,
        model_dir: str = "models/dci",
        monitor_interval: float = 5.0,
        timeout: Optional[float] = None,
    ) -> None:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        self.file_path      = file_path
        self.receiver_host  = receiver_host
        self.receiver_port  = receiver_port
        self.initial_streams = initial_streams
        self.timeout        = timeout

        file_size = os.path.getsize(file_path)

        print(f"\n  TurboLane Sender")
        print(f"  File     : {os.path.abspath(file_path)}")
        print(f"  Size     : {file_size / 1e6:.2f} MB")
        print(f"  Receiver : {receiver_host}:{receiver_port}")
        print(f"  Streams  : {initial_streams} initial  [{min_streams}..{max_streams}]")
        print(f"  Model    : {model_dir}")
        print(f"  RL cycle : every {monitor_interval}s\n")

        # --- Metrics (shared between transfer layer and adapter) ---
        self._metrics = MetricsCollector()

        # --- Transfer session ---
        self._session = TransferSession(
            file_path=file_path,
            receiver_host=receiver_host,
            receiver_port=receiver_port,
            num_streams=initial_streams,
            metrics_collector=self._metrics,
        )

        # --- TurboLane adapter (embedded RL engine) ---
        self._adapter = TurboLaneAdapter(
            metrics=self._metrics,
            stream_count_callback=self._on_stream_count_change,
            interval=monitor_interval,
            mode="dci",
            algorithm="qlearning",
            model_dir=model_dir,
            min_streams=min_streams,
            max_streams=max_streams,
            default_streams=initial_streams,
        )

    # ------------------------------------------------------------------
    # Callback: TurboLane → TransferSession
    # ------------------------------------------------------------------

    def _on_stream_count_change(self, new_count: int) -> None:
        """
        Called by the TurboLane adapter whenever it recommends a different
        number of streams. Forwards to the transfer session.
        """
        self._session.adjust_streams(new_count)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> bool:
        """
        Start the transfer and block until completion or error.

        Returns:
            True if the file was fully transferred, False otherwise.
        """
        start = time.monotonic()

        # Start TurboLane monitoring loop first (it waits one interval before first decision)
        self._adapter.start()

        # Start the transfer
        self._session.start()

        # Block until done
        success = self._session.wait(timeout=self.timeout)

        # Stop the adapter and persist the Q-table
        self._adapter.stop()

        elapsed = time.monotonic() - start
        stats = self._session.get_stats()

        self._print_summary(stats, elapsed, success)
        return success

    def _print_summary(self, stats: dict, elapsed: float, success: bool) -> None:
        status = "✓ COMPLETED" if success else "✗ FAILED"
        error  = f" — {stats['error']}" if stats.get("error") else ""

        print(f"\n  {'─'*56}")
        print(f"  Transfer Summary")
        print(f"  {'─'*56}")
        print(f"  Status          : {status}{error}")
        print(f"  File            : {stats['file_name']}")
        print(f"  Size            : {stats['file_size'] / 1e6:.2f} MB")
        print(f"  Duration        : {elapsed:.2f} s")
        print(f"  Avg throughput  : {stats['throughput_mbps']:.2f} Mbps")
        print(f"  Chunks          : {stats['acked_chunks']}/{stats['total_chunks']}")
        print(f"  Progress        : {stats['progress_pct']}%")

        adapter_stats = self._adapter.get_stats()
        print(f"  RL decisions    : {adapter_stats.get('total_decisions', 0)}")
        print(f"  Q-table states  : {adapter_stats.get('q_table_states', 0)}")
        print(f"  Exploration ε   : {adapter_stats.get('exploration_rate', 0):.4f}")
        print(f"  {'─'*56}\n")
