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
        self._min_connections = min_connections
        self._max_connections = max_connections

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
            discretize_fn=self._discretize_state,
            reward_fn=self._compute_reward,
            constraint_fn=self._apply_constraints,
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
    # DCI policy hooks injected into RLAgent
    # -----------------------------------------------------------------------

    def _discretize_state(
        self,
        throughput_mbps: float,
        rtt_ms: float,
        loss_pct: float,
    ) -> tuple:
        # Throughput level (0-4)
        if throughput_mbps < 10:
            t = 0
        elif throughput_mbps < 50:
            t = 1
        elif throughput_mbps < 100:
            t = 2
        elif throughput_mbps < 500:
            t = 3
        else:
            t = 4

        # RTT level (0-3)
        if rtt_ms < 30:
            r = 0
        elif rtt_ms < 80:
            r = 1
        elif rtt_ms < 150:
            r = 2
        else:
            r = 3

        # Loss level (0-4)
        if loss_pct < 0.1:
            l = 0
        elif loss_pct < 0.5:
            l = 1
        elif loss_pct < 1.0:
            l = 2
        elif loss_pct < 2.0:
            l = 3
        else:
            l = 4

        return (t, r, l)

    def _compute_reward(
        self,
        prev_throughput: float,
        curr_throughput: float,
        curr_loss_pct: float,
        curr_rtt_ms: float,
        num_streams: int,
    ) -> float:
        # Throughput improvement (primary signal from the paper)
        tput_delta = curr_throughput - prev_throughput

        # Quadratic loss penalty (paper: T*L*B term)
        loss_penalty = (curr_loss_pct ** 2) * 0.5

        # RTT penalty — high RTT signals the network is filling up
        rtt_penalty = max(0.0, (curr_rtt_ms - 50.0) * 0.01)

        # Progressive stream overhead penalty (paper: T/Kn term — cost of n)
        if num_streams <= 4:
            stream_penalty = 0.0
        elif num_streams <= 8:
            stream_penalty = (num_streams - 4) * 0.05
        elif num_streams <= 12:
            stream_penalty = 0.2 + (num_streams - 8) * 0.1
        else:
            stream_penalty = 0.6 + (num_streams - 12) * 0.3

        # Stability bonus: reward holding a good operating point
        stability_bonus = (
            0.3
            if abs(tput_delta) < 5.0
            and curr_throughput > 50.0
            and curr_loss_pct < 0.5
            else 0.0
        )

        reward = (
            tput_delta * 0.1      # scale down so it doesn't dominate
            + stability_bonus
            - loss_penalty
            - rtt_penalty
            - stream_penalty
        )

        # Clip for training stability
        return max(-5.0, min(5.0, reward))

    def _apply_constraints(
        self,
        proposed_connections: int,
        current_connections: int,
        recent_metrics: list,
    ) -> int:
        del current_connections, recent_metrics
        return max(self._min_connections, min(self._max_connections, proposed_connections))

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
