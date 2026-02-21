"""
turbolane/rl/agent.py

Q-Learning agent for optimizing parallel TCP stream count.

Design principles:
- State: discretized (throughput_level, rtt_level, loss_level) tuple
- Actions: 5 discrete actions mapping to stream count deltas
- Reward: throughput improvement minus congestion penalties
- No application code. No sockets. No file I/O. Pure RL logic.
"""

import random
import time
import logging
from collections import deque

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action space: index → stream count delta
# Matches paper's design: aggressive +, conservative +, hold, conservative -, aggressive -
# ---------------------------------------------------------------------------
ACTIONS = {
    0: +2,   # aggressive increase
    1: +1,   # conservative increase
    2:  0,   # hold
    3: -1,   # conservative decrease
    4: -2,   # aggressive decrease
}
NUM_ACTIONS = len(ACTIONS)


class RLAgent:
    """
    Q-Learning agent for TCP stream count optimization.

    Public interface (all the rest of the app needs):
        make_decision(throughput, rtt, loss_pct)  → int  (new stream count)
        learn_from_feedback(throughput, rtt, loss_pct)    (update Q-table)
        get_stats()                               → dict
        reset()                                           (clear learned state)
    """

    def __init__(
        self,
        min_connections: int = 1,
        max_connections: int = 16,
        default_connections: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.8,
        exploration_rate: float = 0.3,
        exploration_decay: float = 0.995,
        min_exploration: float = 0.05,
        monitoring_interval: float = 5.0,
    ):
        # Connection bounds
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.current_connections = default_connections

        # RL hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration

        # Monitoring interval (agent self-gates decisions)
        self.monitoring_interval = monitoring_interval
        self._last_decision_time: float = 0.0

        # Q-table: state_tuple → {action_int: q_value}
        self.Q: dict[tuple, dict[int, float]] = {}

        # Transition memory (needed for Q-update on next cycle)
        self._last_state: tuple | None = None
        self._last_action: int | None = None
        self._last_metrics: dict | None = None

        # Rolling history for oscillation detection and constraint logic
        self._action_history: deque = deque(maxlen=10)
        self._metrics_history: deque = deque(maxlen=50)

        # Counters for stats and persistence
        self.total_decisions: int = 0
        self.total_updates: int = 0
        self._total_reward: float = 0.0
        self._positive_rewards: int = 0
        self._negative_rewards: int = 0
        self._throughput_improvements: int = 0

        logger.info(
            "RLAgent init: connections=[%d..%d] default=%d "
            "lr=%.3f γ=%.2f ε=%.2f interval=%.1fs",
            min_connections, max_connections, default_connections,
            learning_rate, discount_factor, exploration_rate, monitoring_interval,
        )

    # -----------------------------------------------------------------------
    # State representation
    # -----------------------------------------------------------------------

    def discretize_state(
        self,
        throughput_mbps: float,
        rtt_ms: float,
        loss_pct: float,
    ) -> tuple:
        """
        Map continuous metrics to a discrete state tuple.

        Throughput bins (Mbps): 0-10, 10-50, 50-100, 100-500, 500+
        RTT bins (ms):          0-30, 30-80, 80-150, 150+
        Loss bins (%):          0-0.1, 0.1-0.5, 0.5-1.0, 1.0-2.0, 2.0+

        These bins are tuned for research-lab / data-center traffic.
        Widen or narrow them in config if your link characteristics differ.
        """
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

    # -----------------------------------------------------------------------
    # Q-table access
    # -----------------------------------------------------------------------

    def _init_state(self, state: tuple) -> None:
        """Ensure a state entry exists in the Q-table."""
        if state not in self.Q:
            self.Q[state] = {a: 0.0 for a in range(NUM_ACTIONS)}

    def _get_q(self, state: tuple, action: int) -> float:
        self._init_state(state)
        return self.Q[state][action]

    def _set_q(self, state: tuple, action: int, value: float) -> None:
        self._init_state(state)
        # Clip to prevent runaway values
        self.Q[state][action] = max(-10.0, min(10.0, value))

    def _best_action(self, state: tuple) -> int:
        """Return the action with the highest Q-value for this state."""
        self._init_state(state)
        return max(self.Q[state], key=self.Q[state].__getitem__)

    def _max_q(self, state: tuple) -> float:
        self._init_state(state)
        return max(self.Q[state].values())

    # -----------------------------------------------------------------------
    # Action selection
    # -----------------------------------------------------------------------

    def choose_action(self, state: tuple) -> int:
        """
        Epsilon-greedy action selection with:
        - Boosted exploration for underexplored states
        - Oscillation damping (alternating increase/decrease → hold)
        """
        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay,
        )

        # Boost exploration for states seen fewer than 3 times
        visit_count = sum(
            1 for m in self._metrics_history
            if m.get("state") == state
        )
        effective_epsilon = (
            min(0.6, self.exploration_rate * 2.0)
            if visit_count < 3
            else self.exploration_rate
        )

        # Explore
        if random.random() < effective_epsilon:
            return random.randrange(NUM_ACTIONS)

        # Exploit — check for oscillation first
        best = self._best_action(state)

        if len(self._action_history) >= 4:
            recent = list(self._action_history)[-4:]
            increasing = {0, 1}
            decreasing = {3, 4}
            oscillating = (
                recent[0] in increasing and recent[1] in decreasing and
                recent[2] in increasing and recent[3] in decreasing
            )
            if oscillating:
                logger.debug("Oscillation detected — forcing hold action")
                return 2  # hold

        return best

    def _apply_action(self, action: int, current: int) -> int:
        """
        Apply action delta to current stream count, clamp to [min, max].
        """
        delta = ACTIONS[action]
        new = current + delta
        return max(self.min_connections, min(self.max_connections, new))

    # -----------------------------------------------------------------------
    # Reward function
    # -----------------------------------------------------------------------

    def _compute_reward(
        self,
        prev_throughput: float,
        curr_throughput: float,
        curr_loss_pct: float,
        curr_rtt_ms: float,
        num_streams: int,
    ) -> float:
        """
        Reward function based on the paper's utility:
            U(n, T, L) = T/Kn  −  T*L*B

        We use the delta form: reward = U_t − U_{t-1}

        Components:
          + throughput improvement
          − packet loss penalty   (quadratic, matches paper's non-linear form)
          − RTT penalty           (congestion onset signal)
          − stream overhead       (progressive cost for unnecessary streams)
          + stability bonus       (reward holding a good operating point)
        """
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

    # -----------------------------------------------------------------------
    # Q-table update (Bellman equation)
    # -----------------------------------------------------------------------

    def _update_q(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
    ) -> None:
        """Standard Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s') − Q(s,a)]"""
        current_q = self._get_q(state, action)
        max_next_q = self._max_q(next_state)
        td_target = reward + self.discount_factor * max_next_q
        td_error = td_target - current_q

        # Adaptive learning rate: learn faster from significant events
        effective_lr = (
            self.learning_rate * 1.5
            if abs(reward) > 1.0
            else self.learning_rate
        )

        new_q = current_q + effective_lr * td_error
        self._set_q(state, action, new_q)
        self.total_updates += 1

        logger.debug(
            "Q-update s=%s a=%d r=%.3f td_err=%.3f new_q=%.3f",
            state, action, reward, td_error, new_q,
        )

    # -----------------------------------------------------------------------
    # Public decision interface
    # -----------------------------------------------------------------------

    def should_decide(self) -> bool:
        """True if the monitoring interval has elapsed."""
        return (time.monotonic() - self._last_decision_time) >= self.monitoring_interval

    def make_decision(
        self,
        throughput_mbps: float,
        rtt_ms: float,
        loss_pct: float,
    ) -> int:
        """
        Make a stream count decision based on current network metrics.

        Returns current_connections unchanged if the monitoring interval
        has not yet elapsed (agent self-gates).

        Args:
            throughput_mbps: Observed throughput in Mbps
            rtt_ms:          Observed round-trip time in milliseconds
            loss_pct:        Observed packet loss in percent (0–100)

        Returns:
            Recommended number of parallel streams (int)
        """
        if not self.should_decide():
            return self.current_connections

        state = self.discretize_state(throughput_mbps, rtt_ms, loss_pct)
        action = self.choose_action(state)
        new_connections = self._apply_action(action, self.current_connections)

        # Store transition for learning on next cycle
        self._last_state = state
        self._last_action = action
        self._last_metrics = {
            "throughput": throughput_mbps,
            "rtt": rtt_ms,
            "loss": loss_pct,
            "connections": self.current_connections,
            "state": state,
        }

        changed = new_connections != self.current_connections
        self.current_connections = new_connections
        self._last_decision_time = time.monotonic()
        self.total_decisions += 1
        self._action_history.append(action)

        logger.info(
            "Decision #%d: streams=%d action=%d(%+d) ε=%.3f state=%s",
            self.total_decisions,
            self.current_connections,
            action,
            ACTIONS[action],
            self.exploration_rate,
            state,
        )

        return self.current_connections

    def learn_from_feedback(
        self,
        throughput_mbps: float,
        rtt_ms: float,
        loss_pct: float,
    ) -> None:
        """
        Update Q-table using the outcome of the previous decision.

        Call this once per monitoring cycle, AFTER make_decision(),
        with the metrics observed AFTER the previous action took effect.

        Args:
            throughput_mbps: Current throughput in Mbps
            rtt_ms:          Current RTT in milliseconds
            loss_pct:        Current packet loss in percent (0–100)
        """
        if self._last_state is None or self._last_action is None:
            # First cycle — nothing to learn from yet; just record baseline
            self._last_metrics = {
                "throughput": throughput_mbps,
                "rtt": rtt_ms,
                "loss": loss_pct,
                "connections": self.current_connections,
                "state": self.discretize_state(throughput_mbps, rtt_ms, loss_pct),
            }
            return

        prev = self._last_metrics
        reward = self._compute_reward(
            prev_throughput=prev["throughput"],
            curr_throughput=throughput_mbps,
            curr_loss_pct=loss_pct,
            curr_rtt_ms=rtt_ms,
            num_streams=self.current_connections,
        )

        next_state = self.discretize_state(throughput_mbps, rtt_ms, loss_pct)
        self._update_q(self._last_state, self._last_action, reward, next_state)

        # Update reward stats
        self._total_reward += reward
        if reward > 0:
            self._positive_rewards += 1
        else:
            self._negative_rewards += 1
        if throughput_mbps > prev["throughput"]:
            self._throughput_improvements += 1

        # Append to rolling history
        self._metrics_history.append({
            "state": self._last_state,
            "action": self._last_action,
            "reward": reward,
            "throughput": throughput_mbps,
            "rtt": rtt_ms,
            "loss": loss_pct,
            "connections": self.current_connections,
        })

    # -----------------------------------------------------------------------
    # Stats and reset
    # -----------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return a snapshot of agent statistics for logging and persistence."""
        avg_reward = (
            self._total_reward / self.total_updates
            if self.total_updates > 0
            else 0.0
        )
        return {
            "q_table_states": len(self.Q),
            "current_connections": self.current_connections,
            "exploration_rate": round(self.exploration_rate, 4),
            "total_decisions": self.total_decisions,
            "total_updates": self.total_updates,
            "average_reward": round(avg_reward, 4),
            "total_reward": round(self._total_reward, 4),
            "positive_rewards": self._positive_rewards,
            "negative_rewards": self._negative_rewards,
            "throughput_improvements": self._throughput_improvements,
            "monitoring_interval": self.monitoring_interval,
        }

    def reset(self) -> None:
        """Clear all learned state. Use between transfer sessions if desired."""
        self.Q.clear()
        self._last_state = None
        self._last_action = None
        self._last_metrics = None
        self._action_history.clear()
        self._metrics_history.clear()
        self.total_decisions = 0
        self.total_updates = 0
        self._total_reward = 0.0
        self._positive_rewards = 0
        self._negative_rewards = 0
        self._throughput_improvements = 0
        self.exploration_rate = self.exploration_rate  # keep current decay point
        logger.info("RLAgent reset: Q-table cleared")
