"""
turbolane/rl/agent.py

Q-Learning agent for optimizing parallel TCP stream count.

Design principles:
- State: discretized (throughput_level, rtt_level, loss_level, stream_level) tuple
- Actions: 5 discrete actions matching paper's design (±5, ±1, hold)
- Reward: computed by FederatedPolicy (throughput-first, positive by default)
- No application code. No sockets. No file I/O. Pure RL logic.

FIXES applied (v2):
  1. Actions updated to match the paper exactly: ±5 (aggressive), ±1 (conservative), 0.
     Old actions were ±2 and ±1 which is too slow for the 1-32 stream range.
     The paper explicitly uses +5 and -5 for fast convergence (40% faster claim).

  2. State is now 4-dimensional: (throughput, rtt, loss, stream_count).
     Old 3D state caused total collapse to 1-3 states across 10 episodes.
     Adding stream_count as a dimension multiplies the state space by 5,
     enabling the agent to distinguish (high_tput, 10_streams) from (high_tput, 25_streams).

  3. Oscillation detection window widened to 6 steps (was 4).
     With ±5 actions the agent moves faster so oscillation is more visible.

  4. Exploration boost threshold raised to 5 visits (was 3).
     With 5x more states each state is visited less often — needs more boost.

  5. Q-value clip widened to [-20, 20] (was [-10, 10]).
     The new reward range is [-3, 3] per step; tighter clips stunted learning.

  6. Adaptive learning rate threshold raised to 0.5 (was 1.0) to catch
     the smaller reward magnitudes that occur during normal positive operation.
"""

import random
import time
import logging
from collections import deque
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action space: index → stream count delta
# Matches paper's design exactly: ±5 (aggressive), ±1 (conservative), 0 (hold)
# "The actions include adding five, one, or no TCP streams, and removing
#  one or five TCP streams." — Section III.A.2
# ---------------------------------------------------------------------------
ACTIONS = {
    0: +5,   # aggressive increase
    1: +1,   # conservative increase
    2:  0,   # hold
    3: -1,   # conservative decrease
    4: -5,   # aggressive decrease
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
        max_connections: int = 32,
        default_connections: int = 6,
        learning_rate: float = 0.25,       # increased from 0.15 — learn faster on stable LAN
        discount_factor: float = 0.9,
        exploration_rate: float = 0.2,     # reduced from 0.4 — less random exploration on LAN
        exploration_decay: float = 0.99,   # faster decay from 0.998 — converge quickly
        min_exploration: float = 0.05,
        monitoring_interval: float = 5.0,
        discretize_fn: Optional[Callable] = None,
        reward_fn: Optional[Callable] = None,
        constraint_fn: Optional[Callable] = None,
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

        # Policy hooks (optional). If not provided, use built-in defaults.
        self._discretize = discretize_fn or self.discretize_state
        self._reward = reward_fn or self._compute_reward
        self._constrain = constraint_fn or self._default_constrain

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
            "RLAgent v2 init: connections=[%d..%d] default=%d "
            "lr=%.3f γ=%.2f ε=%.2f interval=%.1fs actions=%s",
            min_connections, max_connections, default_connections,
            learning_rate, discount_factor, exploration_rate, monitoring_interval,
            {a: ACTIONS[a] for a in ACTIONS},
        )

    # -----------------------------------------------------------------------
    # State representation (fallback — normally overridden by FederatedPolicy)
    # -----------------------------------------------------------------------

    def discretize_state(
        self,
        throughput_mbps: float,
        rtt_ms: float,
        loss_pct: float,
    ) -> tuple:
        """
        Fallback discretisation (used only if no discretize_fn is provided).
        FederatedPolicy always injects its own fine-grained version.

        Throughput bins (Mbps): <50, 50-100, 100-125, 125-150, 150-175, 175+
        RTT bins (ms):          <100, 100-250, 250-450, 450-600, 600+
        Loss bins (%):          <0.1, 0.1-0.5, 0.5-1.5, 1.5+
        Stream bins (count):    <8, 8-14, 14-20, 20-26, 26+
        """
        if throughput_mbps < 50:
            t = 0
        elif throughput_mbps < 100:
            t = 1
        elif throughput_mbps < 125:
            t = 2
        elif throughput_mbps < 150:
            t = 3
        elif throughput_mbps < 175:
            t = 4
        else:
            t = 5

        if rtt_ms < 100:
            r = 0
        elif rtt_ms < 250:
            r = 1
        elif rtt_ms < 450:
            r = 2
        elif rtt_ms < 600:
            r = 3
        else:
            r = 4

        if loss_pct < 0.1:
            l = 0
        elif loss_pct < 0.5:
            l = 1
        elif loss_pct < 1.5:
            l = 2
        else:
            l = 3

        streams = self.current_connections
        if streams < 8:
            s = 0
        elif streams < 14:
            s = 1
        elif streams < 20:
            s = 2
        elif streams < 26:
            s = 3
        else:
            s = 4

        return (t, r, l, s)

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
        # Widened clip range: reward is [-3, 3] per step, multi-step horizon ~15 steps
        self.Q[state][action] = max(-20.0, min(20.0, value))

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
        - Boosted exploration for underexplored states (threshold: 5 visits)
        - Oscillation damping (alternating increase/decrease → hold)
        """
        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay,
        )

        # Boost exploration for states seen fewer than 5 times
        visit_count = sum(
            1 for m in self._metrics_history
            if m.get("state") == state
        )
        effective_epsilon = (
            min(0.7, self.exploration_rate * 2.0)
            if visit_count < 5
            else self.exploration_rate
        )

        # Explore
        if random.random() < effective_epsilon:
            return random.randrange(NUM_ACTIONS)

        # Exploit — check for oscillation first
        best = self._best_action(state)

        # Wider oscillation window (6 steps) to catch ±5 swings
        if len(self._action_history) >= 6:
            recent = list(self._action_history)[-6:]
            increasing = {0, 1}
            decreasing = {3, 4}
            # Detect alternating increase/decrease pattern
            oscillating = all(
                (recent[i] in increasing and recent[i + 1] in decreasing) or
                (recent[i] in decreasing and recent[i + 1] in increasing)
                for i in range(len(recent) - 1)
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
        proposed = current + delta
        recent = (
            list(self._metrics_history)[-3:]
            if len(self._metrics_history) >= 3
            else []
        )
        return self._constrain(proposed, current, recent)

    def _default_constrain(
        self,
        proposed_connections: int,
        current_connections: int,
        recent_metrics: list,
    ) -> int:
        del current_connections, recent_metrics
        return max(self.min_connections, min(self.max_connections, proposed_connections))

    # -----------------------------------------------------------------------
    # Fallback reward function (used only if no reward_fn injected)
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
        Fallback reward: simple throughput delta minus loss penalty.
        FederatedPolicy always injects the full reward function instead.
        """
        tput_delta = curr_throughput - prev_throughput
        loss_penalty = curr_loss_pct ** 2 * 0.5
        reward = tput_delta * 0.05 - loss_penalty
        return max(-3.0, min(3.0, reward))

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

        # Adaptive learning rate: learn faster from moderate events
        # Threshold lowered to 0.5 (from 1.0) to catch normal positive rewards
        effective_lr = (
            self.learning_rate * 1.5
            if abs(reward) > 0.5
            else self.learning_rate
        )

        new_q = current_q + effective_lr * td_error
        self._set_q(state, action, new_q)
        self.total_updates += 1

        logger.debug(
            "Q-update s=%s a=%d(%+d) r=%.3f td_err=%.3f new_q=%.3f",
            state, action, ACTIONS[action], reward, td_error, new_q,
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

        state = self._discretize(throughput_mbps, rtt_ms, loss_pct)
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
                "state": self._discretize(throughput_mbps, rtt_ms, loss_pct),
            }
            return

        prev = self._last_metrics
        reward = self._reward(
            prev["throughput"],
            throughput_mbps,
            loss_pct,
            rtt_ms,
            self.current_connections,
        )

        next_state = self._discretize(throughput_mbps, rtt_ms, loss_pct)
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
        logger.info("RLAgent reset: Q-table cleared")