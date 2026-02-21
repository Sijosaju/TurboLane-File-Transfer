"""
turbolane/policies/federated.py

FederatedPolicy — the policy wrapper for data center / DCI environments.

Responsibilities:
- Own the RLAgent instance
- Own the QTableStorage instance
- Expose a clean, stable interface to the engine
- Handle persistence (auto-save, load on init)

Design principles:
- No networking code
- No application logic
- Agent-type-agnostic interface (same methods regardless of Q-learning or PPO)
  This means swapping the backend to PPO only touches __init__, nothing else.
"""

import logging
from pathlib import Path

from turbolane.rl.agent import RLAgent
from turbolane.rl.storage import QTableStorage

logger = logging.getLogger(__name__)


class FederatedPolicy:
    """
    Policy for federated / data-center interconnect (DCI) environments.

    Public interface:
        decide(throughput, rtt, loss_pct)        → int  (stream count)
        learn(throughput, rtt, loss_pct)                (Q-update)
        save()                                          (persist to disk)
        get_stats()                              → dict
        reset()                                         (clear learned state)

    All four methods have identical signatures regardless of whether
    the backend is Q-learning or PPO — future migration is transparent.
    """

    def __init__(
        self,
        model_dir: str = "models/dci",
        min_connections: int = 1,
        max_connections: int = 16,
        default_connections: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.8,
        exploration_rate: float = 0.3,
        exploration_decay: float = 0.995,
        min_exploration: float = 0.05,
        monitoring_interval: float = 5.0,
        auto_save_every: int = 100,    # auto-save every N learning updates
    ):
        """
        Initialize FederatedPolicy with a Q-learning backend.

        Args:
            model_dir:           Directory for Q-table persistence
            min_connections:     Minimum allowed parallel streams
            max_connections:     Maximum allowed parallel streams
            default_connections: Starting stream count
            learning_rate:       Q-learning α
            discount_factor:     Q-learning γ
            exploration_rate:    Initial ε for epsilon-greedy
            exploration_decay:   ε decay per decision
            min_exploration:     Floor for ε
            monitoring_interval: Seconds between decisions (agent self-gates)
            auto_save_every:     Persist Q-table every N updates (0 = off)
        """
        self._auto_save_every = auto_save_every

        # --- Storage (owns the file paths) ---
        self._storage = QTableStorage(model_dir=model_dir)

        # --- Agent (owns the Q-table and RL logic) ---
        self._agent = RLAgent(
            min_connections=min_connections,
            max_connections=max_connections,
            default_connections=default_connections,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            exploration_rate=exploration_rate,
            exploration_decay=exploration_decay,
            min_exploration=min_exploration,
            monitoring_interval=monitoring_interval,
        )

        # Load any previously saved Q-table
        self._load()

        logger.info(
            "FederatedPolicy ready: model_dir=%s connections=[%d..%d]",
            model_dir, min_connections, max_connections,
        )

    # -----------------------------------------------------------------------
    # Core interface
    # -----------------------------------------------------------------------

    def decide(
        self,
        throughput_mbps: float,
        rtt_ms: float,
        loss_pct: float,
    ) -> int:
        """
        Make a stream count decision.

        Args:
            throughput_mbps: Observed throughput in Mbps
            rtt_ms:          Observed RTT in milliseconds
            loss_pct:        Observed packet loss in percent (0–100)

        Returns:
            Recommended number of parallel streams (int)
        """
        return self._agent.make_decision(throughput_mbps, rtt_ms, loss_pct)

    def learn(
        self,
        throughput_mbps: float,
        rtt_ms: float,
        loss_pct: float,
    ) -> None:
        """
        Update Q-table from the outcome of the previous decision.

        Call this once per monitoring cycle, AFTER decide(), with
        the metrics observed after the previous action took effect.

        Args:
            throughput_mbps: Current throughput in Mbps
            rtt_ms:          Current RTT in milliseconds
            loss_pct:        Current packet loss in percent (0–100)
        """
        self._agent.learn_from_feedback(throughput_mbps, rtt_ms, loss_pct)

        # Auto-save check
        if (
            self._auto_save_every > 0
            and self._agent.total_updates > 0
            and self._agent.total_updates % self._auto_save_every == 0
        ):
            logger.debug("Auto-save triggered at update #%d", self._agent.total_updates)
            self.save()

    def save(self) -> bool:
        """
        Persist the current Q-table and stats to disk.

        Returns:
            True on success, False on failure.
        """
        return self._storage.save(self._agent.Q, self._agent.get_stats())

    def get_stats(self) -> dict:
        """Return agent statistics for monitoring and logging."""
        stats = self._agent.get_stats()
        stats["model_dir"] = self._storage.model_dir
        stats["model_exists_on_disk"] = self._storage.exists()
        return stats

    def reset(self) -> None:
        """
        Clear the agent's learned state (Q-table, history, counters).
        Does NOT delete files on disk — call storage.delete() for that.
        """
        self._agent.reset()
        logger.info("FederatedPolicy: agent state reset")

    # -----------------------------------------------------------------------
    # Properties (read-only access for adapter if needed)
    # -----------------------------------------------------------------------

    @property
    def current_connections(self) -> int:
        return self._agent.current_connections

    @property
    def agent(self) -> RLAgent:
        """
        Direct access to the underlying agent.
        Used by the adapter for state discretization.
        Avoid accessing this from application code.
        """
        return self._agent

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _load(self) -> None:
        """Load Q-table from disk into the agent."""
        Q, metadata = self._storage.load()
        if Q:
            self._agent.Q = Q
            # Restore exploration rate from metadata if available
            saved_epsilon = metadata.get("exploration_rate")
            if saved_epsilon is not None:
                self._agent.exploration_rate = max(
                    self._agent.min_exploration,
                    float(saved_epsilon),
                )
            # Restore counters
            self._agent.total_decisions = int(metadata.get("total_decisions", 0))
            self._agent.total_updates   = int(metadata.get("total_updates", 0))
            logger.info(
                "Restored: %d Q-states, %d decisions, ε=%.4f",
                len(Q),
                self._agent.total_decisions,
                self._agent.exploration_rate,
            )

    def __repr__(self) -> str:
        return (
            f"FederatedPolicy("
            f"connections={self.current_connections}, "
            f"q_states={len(self._agent.Q)}, "
            f"ε={self._agent.exploration_rate:.4f})"
        )
