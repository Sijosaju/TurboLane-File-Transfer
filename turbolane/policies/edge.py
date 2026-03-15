"""
turbolane/policies/edge.py

EdgePolicy — the policy wrapper for edge / public internet environments.
"""

import logging
from pathlib import Path

from turbolane.rl.agent import RLAgent
from turbolane.rl.storage import QTableStorage

logger = logging.getLogger(__name__)


class EdgePolicy:
    """
    Policy for edge / public internet download environments.

    Public interface:
        decide(throughput, rtt, loss_pct)        → int  (stream count)
        learn(throughput, rtt, loss_pct)                (Q-update)
        save()                                          (persist to disk)
        get_stats()                              → dict
        reset()                                         (clear learned state)
    """

    # Optimal stream range — tuned for good connections (low RTT, low loss)
    OPTIMAL_MIN = 8        # was 6 — floor raised for better throughput
    OPTIMAL_MAX = 16       # was 10 — allow full 16 streams on good connections
    OPTIMAL_BONUS = 20.0   # was 12.0 — stronger incentive to stay high
    EXTENDED_MAX = 16      # was 12
    EXTENDED_BONUS = 10.0  # was 5.0

    def __init__(
        self,
        model_dir: str = "models/edge",
        min_connections: int = 1,
        max_connections: int = 16,
        default_connections: int = 8,
        learning_rate: float = 0.1,
        discount_factor: float = 0.8,
        exploration_rate: float = 0.3,
        exploration_decay: float = 0.995,
        min_exploration: float = 0.05,
        monitoring_interval: float = 5.0,
        auto_save_every: int = 50,
    ):
        self._auto_save_every = auto_save_every
        self._min_connections = min_connections
        self._max_connections = max_connections

        self._storage = QTableStorage(model_dir=model_dir)

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

        self._load()

        logger.info(
            "EdgePolicy ready: model_dir=%s connections=[%d..%d] optimal=[%d..%d]",
            model_dir, min_connections, max_connections,
            self.OPTIMAL_MIN, self.OPTIMAL_MAX,
        )

    # -----------------------------------------------------------------------
    # Core interface
    # -----------------------------------------------------------------------

    def decide(self, throughput_mbps: float, rtt_ms: float, loss_pct: float) -> int:
        return self._agent.make_decision(throughput_mbps, rtt_ms, loss_pct)

    def learn(self, throughput_mbps: float, rtt_ms: float, loss_pct: float) -> None:
        self._agent.learn_from_feedback(throughput_mbps, rtt_ms, loss_pct)

        if (
            self._auto_save_every > 0
            and self._agent.total_updates > 0
            and self._agent.total_updates % self._auto_save_every == 0
        ):
            logger.debug("Auto-save triggered at update #%d", self._agent.total_updates)
            self.save()

    def save(self) -> bool:
        return self._storage.save(self._agent.Q, self._agent.get_stats())

    def get_stats(self) -> dict:
        stats = self._agent.get_stats()
        stats["model_dir"] = self._storage.model_dir
        stats["model_exists_on_disk"] = self._storage.exists()
        connections = self._agent.current_connections
        if self.OPTIMAL_MIN <= connections <= self.OPTIMAL_MAX:
            stats["stream_range_status"] = "optimal"
        elif connections <= self.EXTENDED_MAX:
            stats["stream_range_status"] = "extended"
        else:
            stats["stream_range_status"] = "above_optimal"
        return stats

    def reset(self) -> None:
        self._agent.reset()
        logger.info("EdgePolicy: agent state reset")

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def current_connections(self) -> int:
        return self._agent.current_connections

    @property
    def agent(self) -> RLAgent:
        return self._agent

    # -----------------------------------------------------------------------
    # Edge-specific policy functions (injected into RLAgent)
    # -----------------------------------------------------------------------

    def _discretize_state(
        self,
        throughput_mbps: float,
        rtt_ms: float,
        loss_pct: float,
    ) -> tuple:
        """
        Edge-tuned state discretization.

        Throughput bins (Mbps): 0-10, 10-20, 20-30, 30-40, 40-50, 50+
        RTT bins (ms): 0-50, 50-150, 150-300, 300-600, 600-1000, 1000+
        Loss bins (%): 0-0.1, 0.1-0.5, 0.5-1.0, 1.0-2.0, 2.0+

        Total states: 6 × 6 × 5 = 180
        """
        # Throughput level (0-5)
        if throughput_mbps < 10:
            t = 0
        elif throughput_mbps < 20:
            t = 1
        elif throughput_mbps < 30:
            t = 2
        elif throughput_mbps < 40:
            t = 3
        elif throughput_mbps < 50:
            t = 4
        else:
            t = 5

        # RTT level (0-5)
        if rtt_ms < 50:
            r = 0
        elif rtt_ms < 150:
            r = 1
        elif rtt_ms < 300:
            r = 2
        elif rtt_ms < 600:
            r = 3
        elif rtt_ms < 1000:
            r = 4
        else:
            r = 5

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
        """
        Edge reward function — tuned to maximize download speed.

        Changes from previous version:
        - Loss penalty reduced (0.10-0.30% is normal, not worth penalizing)
        - RTT penalty threshold raised to 100ms (10ms should have zero penalty)
        - Added absolute throughput bonus (rewards high speed, not just delta)
        - Stream penalty removed inside optimal range
        - Stronger optimal range bonus
        """
        # Throughput improvement delta
        tput_delta = curr_throughput - prev_throughput

        # Light loss penalty — normal internet loss (< 0.5%) barely penalized
        loss_penalty = (curr_loss_pct ** 2) * 0.1   # was 0.5

        # RTT penalty only kicks in above 100ms — 10ms RTT = zero penalty
        rtt_penalty = max(0.0, (curr_rtt_ms - 100.0) * 0.002)  # was 50ms / 0.005

        # Stream penalty — only outside optimal range
        if num_streams <= self.OPTIMAL_MAX:
            stream_penalty = 0.0
        else:
            stream_penalty = (num_streams - self.OPTIMAL_MAX) * 0.2

        # Absolute throughput bonus — rewards staying at high speed
        throughput_bonus = min(3.0, curr_throughput * 0.05)

        # Efficiency bonus: throughput per stream
        efficiency = curr_throughput / max(1, num_streams)
        efficiency_bonus = min(2.0, efficiency * 0.1) if efficiency > 4.0 else 0.0

        reward = (
            tput_delta * 0.2          # was 0.1
            + throughput_bonus        # new
            + efficiency_bonus
            - loss_penalty
            - rtt_penalty
            - stream_penalty
        )

        # Strong bonus for staying in optimal range
        if self.OPTIMAL_MIN <= num_streams <= self.OPTIMAL_MAX:
            reward += self.OPTIMAL_BONUS * 0.15   # was 0.1
        elif num_streams <= self.EXTENDED_MAX:
            reward += self.EXTENDED_BONUS * 0.1

        return max(-5.0, min(5.0, reward))

    def _apply_constraints(
        self,
        proposed_connections: int,
        current_connections: int,
        recent_metrics: list,
    ) -> int:
        """
        Edge-specific action constraints.

        Key change: hard floor at OPTIMAL_MIN on good connections —
        agent can never drop below 8 streams when network is healthy.
        """
        result = max(self._min_connections, min(self._max_connections, proposed_connections))

        if not recent_metrics:
            return result

        avg_throughput = sum(m["throughput"] for m in recent_metrics) / len(recent_metrics)
        avg_loss = sum(m["loss"] for m in recent_metrics) / len(recent_metrics)
        avg_rtt = sum(m["rtt"] for m in recent_metrics) / len(recent_metrics)

        # Good conditions — hard floor at OPTIMAL_MIN, never drop below it
        if avg_throughput > 5 and avg_loss < 1.0 and avg_rtt < 200:
            if result < self.OPTIMAL_MIN:
                logger.debug("EdgePolicy: good conditions, hard floor at OPTIMAL_MIN=%d", self.OPTIMAL_MIN)
                return self.OPTIMAL_MIN

        # Poor conditions: limit increases to +1 at a time
        if avg_loss > 2.0 or avg_rtt > 1000:
            if proposed_connections > current_connections:
                logger.debug("EdgePolicy: poor conditions, limiting increase to +1")
                return min(current_connections + 1, result)

        return result

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _load(self) -> None:
        Q, metadata = self._storage.load()
        if Q:
            self._agent.Q = Q

            # Restore exploration rate
            saved_epsilon = metadata.get("exploration_rate")
            if saved_epsilon is not None:
                self._agent.exploration_rate = max(
                    self._agent.min_exploration,
                    float(saved_epsilon),
                )

            # Restore decision/update counters
            self._agent.total_decisions = int(metadata.get("total_decisions", 0))
            self._agent.total_updates = int(metadata.get("total_updates", 0))

            # Restore reward stats
            self._agent._total_reward = float(metadata.get("total_reward", 0.0))
            self._agent._positive_rewards = int(metadata.get("positive_rewards", 0))
            self._agent._negative_rewards = int(metadata.get("negative_rewards", 0))
            self._agent._throughput_improvements = int(metadata.get("throughput_improvements", 0))

            logger.info(
                "EdgePolicy restored: %d Q-states, %d decisions, %d updates, ε=%.4f",
                len(Q),
                self._agent.total_decisions,
                self._agent.total_updates,
                self._agent.exploration_rate,
            )

    def __repr__(self) -> str:
        return (
            f"EdgePolicy("
            f"connections={self.current_connections}, "
            f"q_states={len(self._agent.Q)}, "
            f"ε={self._agent.exploration_rate:.4f})"
        )