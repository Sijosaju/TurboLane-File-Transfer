"""
turbolane/engine.py

TurboLaneEngine — the single public entry point for the TurboLane SDK.

This is the ONLY class that application code (or the adapter) should import.
All policy selection, agent wiring, and mode routing happens here.

Supported modes:
    'dci'    → FederatedPolicy (data center / private network) ← current use
    'client' → EdgePolicy      (public internet / edge)         ← future

Supported algorithms:
    'qlearning' → RLAgent      ← current use
    'ppo'       → PPOAgent     ← future (drop-in via FederatedPolicy)

Usage (DCI + Q-learning):
    from turbolane.engine import TurboLaneEngine

    engine = TurboLaneEngine(mode='dci', algorithm='qlearning')
    streams = engine.decide(throughput_mbps, rtt_ms, loss_pct)
    engine.learn(throughput_mbps, rtt_ms, loss_pct)
    engine.save()
"""

import logging

logger = logging.getLogger(__name__)

# Supported modes and their policy classes
_MODE_POLICY_MAP = {
    "dci":    "turbolane.policies.federated.FederatedPolicy",
    "client": "turbolane.policies.edge.EdgePolicy",        # future
}

_VALID_ALGORITHMS = {"qlearning", "ppo"}


class TurboLaneEngine:
    """
    Unified TurboLane control-plane engine.

    Public interface (identical regardless of mode or algorithm):
        decide(throughput_mbps, rtt_ms, loss_pct)  → int
        learn(throughput_mbps, rtt_ms, loss_pct)
        save()                                      → bool
        get_stats()                                 → dict
        reset()

    Convenience properties:
        .current_connections                        → int
        .mode                                       → str
        .algorithm                                  → str
    """

    def __init__(
        self,
        mode: str = "dci",
        algorithm: str = "qlearning",
        **policy_kwargs,
    ):
        """
        Initialize TurboLane engine.

        Args:
            mode:          'dci' or 'client'
            algorithm:     'qlearning' or 'ppo'
            **policy_kwargs: Passed directly to the policy constructor.
                           See FederatedPolicy.__init__ for valid keys.

        Example:
            TurboLaneEngine(
                mode='dci',
                algorithm='qlearning',
                model_dir='models/dci',
                min_connections=1,
                max_connections=32,
                default_connections=4,
                monitoring_interval=5.0,
            )
        """
        mode = mode.lower()
        algorithm = algorithm.lower().replace("-", "").replace("_", "")

        if mode not in _MODE_POLICY_MAP:
            raise ValueError(
                f"Unknown mode '{mode}'. Valid modes: {list(_MODE_POLICY_MAP)}"
            )
        if algorithm not in _VALID_ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. Valid: {list(_VALID_ALGORITHMS)}"
            )

        self.mode = mode
        self.algorithm = algorithm

        # Instantiate the appropriate policy
        self._policy = self._build_policy(mode, algorithm, policy_kwargs)

        logger.info(
            "TurboLaneEngine ready: mode=%s algorithm=%s",
            self.mode, self.algorithm,
        )

    # -----------------------------------------------------------------------
    # Core interface — these are the ONLY methods the adapter calls
    # -----------------------------------------------------------------------

    def decide(
        self,
        throughput_mbps: float,
        rtt_ms: float,
        loss_pct: float,
    ) -> int:
        """
        Make a stream count recommendation based on current network metrics.

        Args:
            throughput_mbps: Observed throughput in Mbps
            rtt_ms:          Observed round-trip time in milliseconds
            loss_pct:        Observed packet loss in percent (0–100)

        Returns:
            Recommended number of parallel TCP streams (int)
        """
        return self._policy.decide(throughput_mbps, rtt_ms, loss_pct)

    def learn(
        self,
        throughput_mbps: float,
        rtt_ms: float,
        loss_pct: float,
    ) -> None:
        """
        Update the policy from the outcome of the previous decision.

        Call this once per monitoring cycle, AFTER decide(), with
        the metrics observed after the previous action took effect.

        Args:
            throughput_mbps: Current throughput in Mbps
            rtt_ms:          Current RTT in milliseconds
            loss_pct:        Current packet loss in percent (0–100)
        """
        self._policy.learn(throughput_mbps, rtt_ms, loss_pct)

    def save(self) -> bool:
        """
        Persist the policy to disk.

        Returns:
            True on success, False on failure.
        """
        return self._policy.save()

    def get_stats(self) -> dict:
        """Return a stats dict for logging, monitoring, and CLI display."""
        stats = self._policy.get_stats()
        stats["engine_mode"] = self.mode
        stats["engine_algorithm"] = self.algorithm
        return stats

    def reset(self) -> None:
        """Clear the policy's learned state. Does not delete files on disk."""
        self._policy.reset()

    # -----------------------------------------------------------------------
    # Convenience properties
    # -----------------------------------------------------------------------

    @property
    def current_connections(self) -> int:
        """Current recommended stream count."""
        return self._policy.current_connections

    # -----------------------------------------------------------------------
    # Internal factory
    # -----------------------------------------------------------------------

    def _build_policy(self, mode: str, algorithm: str, kwargs: dict):
        """
        Instantiate the correct policy for the given mode.

        DCI mode: always uses FederatedPolicy, passes algorithm as a hint
        for future PPO support. Currently FederatedPolicy only supports
        Q-learning; PPO support will be added to FederatedPolicy directly.
        """
        if mode == "dci":
            from turbolane.policies.federated import FederatedPolicy
            return FederatedPolicy(**kwargs)

        elif mode == "client":
            # Future: EdgePolicy
            raise NotImplementedError(
                "Client/Edge mode is not yet implemented. "
                "Use mode='dci' for data center transfers."
            )

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __repr__(self) -> str:
        return (
            f"TurboLaneEngine("
            f"mode={self.mode!r}, "
            f"algorithm={self.algorithm!r}, "
            f"connections={self.current_connections})"
        )
