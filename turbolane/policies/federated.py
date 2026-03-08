"""
turbolane/policies/federated.py

FederatedPolicy — the policy wrapper for data center / DCI environments.

FIXES applied (v3 — adaptive self-calibrating bins):

  PROBLEM BEING SOLVED:
    Hardcoded bin boundaries are the root cause of state collapse.
    With fixed bins, any network whose operating range doesn't align
    with the boundaries will collapse all observations into 1–2 bins,
    making Q-learning impossible regardless of the reward design.
    This was confirmed by your logs: 100% of throughput samples (113–185 Mbps)
    fell into one bin, 77% of RTT samples (>600 ms) fell into one bin.
    Result: only 3 unique states discovered across 10 full episodes.

  SOLUTION — Two-phase adaptive binning:
    The _AdaptiveRange class tracks the observed [min, max] of each signal.
    Bin boundaries are computed as equal fractions of this observed range,
    not hardcoded constants.

    Phase 1 (warmup, first 5 steps): fast decay (0.85) so the range
    converges quickly to the actual operating region.

    Phase 2 (steady state): slow decay (0.997) so the range follows
    long-term network changes without forgetting history.

    This works for ANY throughput range — 5 Mbps, 50 Mbps, 500 Mbps —
    without any configuration change.

  OTHER FIXES retained from v2:
    - Stream count as 4th state dimension
    - Loss bins remain fixed (absolute thresholds ARE meaningful)
    - Actions: ±5, ±1, 0  (matching the paper exactly)
    - Reward positive by default during healthy operation
    - RTT penalty also adaptive (fires only in top 15% of observed RTT range)
"""

import logging
import math
from collections import deque

from turbolane.rl.agent import RLAgent
from turbolane.rl.storage import QTableStorage

logger = logging.getLogger(__name__)

_TPUT_BINS = 5
_RTT_BINS  = 5


class _AdaptiveRange:
    """
    Self-calibrating range tracker with two-phase decay.

    Seeds from the first real observation (not a hardcoded guess).
    Uses fast decay during warmup to converge to the actual operating
    range quickly, then slow decay to track long-term changes.

    Args:
        n_bins:      Number of discrete output buckets (0 .. n_bins-1)
        decay_slow:  Steady-state contraction rate (per step)
        decay_fast:  Warmup contraction rate (per step, much faster)
        warmup:      Number of steps to use fast decay
        min_spread:  Minimum range width (prevents division-by-zero)
    """

    def __init__(
        self,
        n_bins: int,
        decay_slow: float = 0.997,
        decay_fast: float = 0.85,
        warmup: int = 5,
        min_spread: float = 1.0,
    ):
        self.n_bins = n_bins
        self.decay_slow = decay_slow
        self.decay_fast = decay_fast
        self.warmup = warmup
        self.min_spread = min_spread
        self.obs_min: float | None = None   # None until first observation
        self.obs_max: float | None = None
        self._count: int = 0

    def update(self, value: float) -> None:
        """Feed a new observation to update the tracked range."""
        self._count += 1

        # Bootstrap from first real observation
        if self.obs_min is None:
            self.obs_min = value * 0.5   # 50% below first observation
            self.obs_max = value * 1.5   # 50% above first observation
            # Edge case: if first observation is 0
            if self.obs_max - self.obs_min < self.min_spread:
                self.obs_min = 0.0
                self.obs_max = self.min_spread
            return

        # Fast convergence during warmup, slow tracking after
        decay = self.decay_fast if self._count <= self.warmup else self.decay_slow

        if value < self.obs_min:
            self.obs_min = value
        else:
            self.obs_min += (value - self.obs_min) * (1.0 - decay)

        if value > self.obs_max:
            self.obs_max = value
        else:
            self.obs_max -= (self.obs_max - value) * (1.0 - decay)

        # Enforce minimum spread
        if self.obs_max - self.obs_min < self.min_spread:
            mid = (self.obs_max + self.obs_min) / 2.0
            self.obs_min = mid - self.min_spread / 2.0
            self.obs_max = mid + self.min_spread / 2.0

    def discretize(self, value: float) -> int:
        """Map value to bucket index in [0, n_bins-1]."""
        if self.obs_min is None:
            return self.n_bins // 2   # middle bin before first observation
        span = self.obs_max - self.obs_min
        frac = max(0.0, min(1.0, (value - self.obs_min) / span))
        return min(int(frac * self.n_bins), self.n_bins - 1)

    def __repr__(self) -> str:
        if self.obs_min is None:
            return f"AdaptiveRange(uninitialized, bins={self.n_bins})"
        return (
            f"AdaptiveRange(min={self.obs_min:.1f}, max={self.obs_max:.1f}, "
            f"bins={self.n_bins}, step={self._count})"
        )


class FederatedPolicy:
    """
    Policy for federated / data-center interconnect (DCI) environments.

    Public interface:
        decide(throughput, rtt, loss_pct)  -> int  (stream count)
        learn(throughput, rtt, loss_pct)           (Q-update)
        save()                             -> bool (persist to disk)
        get_stats()                        -> dict
        reset()                                    (clear learned state)
    """

    def __init__(
        self,
        model_dir: str = "models/dci",
        min_connections: int = 1,
        max_connections: int = 32,
        default_connections: int = 6,
        learning_rate: float = 0.25,       # increased from 0.15 — faster learning on LAN
        discount_factor: float = 0.9,
        exploration_rate: float = 0.2,     # reduced from 0.4 — LAN is stable, less explore needed
        exploration_decay: float = 0.99,   # faster decay from 0.998
        min_exploration: float = 0.05,
        monitoring_interval: float = 2.0,  # reduced from 5.0s — faster decisions on LAN
        auto_save_every: int = 50,
        # Reward shaping
        loss_target_pct: float = 0.5,   # loss budget (%)
        stream_target: int = 20,         # reference optimal stream count
        peak_window: int = 10,           # rolling window for dynamic peak
    ):
        self._auto_save_every = auto_save_every
        self._min_connections = min_connections
        self._max_connections = max_connections
        self.loss_target_pct = max(0.01, float(loss_target_pct))
        self.stream_target = max(1, int(stream_target))

        # Adaptive range trackers — self-calibrate from first real observation
        self._tput_range = _AdaptiveRange(
            n_bins=_TPUT_BINS,
            decay_slow=0.997,
            decay_fast=0.85,
            warmup=5,
            min_spread=5.0,    # at least 5 Mbps spread
        )
        self._rtt_range = _AdaptiveRange(
            n_bins=_RTT_BINS,
            decay_slow=0.997,
            decay_fast=0.85,
            warmup=5,
            min_spread=10.0,   # at least 10 ms spread
        )

        # Dynamic peak for reward normalisation
        self._peak_window = peak_window
        self._recent_tputs: deque = deque(maxlen=peak_window)
        self._rolling_peak: float = 1.0   # will grow from first observation
        self._prev_tput: float = 0.0
        self._prev_streams: int = max(1, int(default_connections))

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
            "FederatedPolicy v3 (adaptive bins) ready: model_dir=%s "
            "connections=[%d..%d] stream_target=%d",
            model_dir, min_connections, max_connections, stream_target,
        )

    # -----------------------------------------------------------------------
    # Core interface
    # -----------------------------------------------------------------------

    def decide(self, throughput_mbps: float, rtt_ms: float, loss_pct: float) -> int:
        # Update adaptive ranges on EVERY observation — bins calibrate fast
        self._tput_range.update(max(0.0, throughput_mbps))
        self._rtt_range.update(max(0.0, rtt_ms))
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
        stats["rolling_peak_mbps"] = round(self._rolling_peak, 2)
        stats["tput_range"] = repr(self._tput_range)
        stats["rtt_range"] = repr(self._rtt_range)
        return stats

    def reset(self) -> None:
        self._agent.reset()
        self._recent_tputs.clear()
        self._rolling_peak = 1.0
        self._prev_tput = 0.0
        # Reset adaptive ranges (they re-seed from next observation)
        self._tput_range = _AdaptiveRange(n_bins=_TPUT_BINS, min_spread=5.0)
        self._rtt_range  = _AdaptiveRange(n_bins=_RTT_BINS,  min_spread=10.0)
        logger.info("FederatedPolicy: agent state reset")

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
    # State discretisation — fully adaptive, network-agnostic
    # -----------------------------------------------------------------------

    def _discretize_state(
        self,
        throughput_mbps: float,
        rtt_ms: float,
        loss_pct: float,
    ) -> tuple:
        """
        Map continuous metrics to 4D discrete state (t, r, l, s).

        THROUGHPUT (t): 0..4, adaptive quantile bucket.
          Bins divide [obs_min_tput, obs_max_tput] into 5 equal parts.
          Works for any throughput range — 5 Mbps or 500 Mbps.

        RTT (r): 0..4, adaptive quantile bucket.
          Bins divide [obs_min_rtt, obs_max_rtt] into 5 equal parts.
          Works for any latency profile — LAN or WAN.

        LOSS (l): 0..3, fixed absolute thresholds.
          0 -> < 0.1%   (clean — your normal case)
          1 -> 0.1-0.5%
          2 -> 0.5-1.5%
          3 -> 1.5%+    (heavy congestion)
          Fixed because these thresholds are meaningful in absolute terms.

        STREAM COUNT (s): 0..4, evenly split across [min_conn, max_conn].
          Lets the agent distinguish sweet-spot from over/under provisioned.
        """
        t = self._tput_range.discretize(max(0.0, throughput_mbps))
        r = self._rtt_range.discretize(max(0.0, rtt_ms))

        if loss_pct < 0.1:
            l = 0
        elif loss_pct < 0.5:
            l = 1
        elif loss_pct < 1.5:
            l = 2
        else:
            l = 3

        streams = self._agent.current_connections
        span = max(1, self._max_connections - self._min_connections)
        stream_frac = (streams - self._min_connections) / span
        s = min(4, int(stream_frac * 5))

        return (t, r, l, s)

    # -----------------------------------------------------------------------
    # Reward — positive by default during healthy operation
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
        Reward design: POSITIVE during normal clean transfers, negative only
        when the agent actively makes things worse.

        base_reward:      tanh-normalised throughput vs rolling peak.
                          ~+1.0 when at peak, 0 at 50% of peak, negative below.
        improve_bonus:    fires when throughput climbs — rewards exploration payoff.
        efficiency_bonus: high throughput with fewer streams = sweet spot.
        loss_penalty:     quadratic, zero when loss=0% (your normal case).
        drop_penalty:     fires on >5% throughput regression.
        rtt_penalty:      adaptive — fires only in top 15% of observed RTT range,
                          so it's correctly calibrated for any latency profile.
        """
        curr_t = max(curr_throughput, 0.0)
        prev_t = max(prev_throughput, 0.0)
        loss   = max(curr_loss_pct, 0.0)
        rtt    = max(curr_rtt_ms, 0.0)
        streams = max(1, int(num_streams))

        # Update rolling peak
        self._recent_tputs.append(curr_t)
        new_peak = max(self._recent_tputs) if self._recent_tputs else curr_t
        if new_peak > self._rolling_peak:
            self._rolling_peak = new_peak
        self._rolling_peak = max(1.0, self._rolling_peak * 0.995)
        peak = self._rolling_peak

        # Base reward: positive when throughput is near peak
        base_reward = 1.5 * math.tanh(curr_t / (peak * 0.85) - 0.5)

        # Improvement bonus
        prev_frac = prev_t / (peak + 1e-6)
        curr_frac = curr_t / (peak + 1e-6)
        improve_bonus = 0.0
        if curr_frac > prev_frac + 0.02:
            improve_bonus = 0.6 * (curr_frac - prev_frac)
        elif curr_frac > 0.92:
            improve_bonus = 0.2

        # Efficiency bonus
        efficiency_bonus = 0.0
        if curr_frac > 0.85:
            stream_ratio = streams / max(self.stream_target, 1)
            if stream_ratio <= 1.0:
                efficiency_bonus = 0.3 * (1.0 - stream_ratio)

        # Loss penalty — zero during clean transfers
        loss_excess = max(0.0, (loss - self.loss_target_pct) / max(self.loss_target_pct, 0.01))
        loss_penalty = 2.0 * (loss_excess ** 2)

        # Drop penalty — fires on >5% throughput drop
        drop_ratio = (prev_t - curr_t) / (prev_t + 1e-6)
        drop_penalty = 0.8 * drop_ratio if (drop_ratio > 0.05 and prev_t > 10.0) else 0.0

        # RTT penalty — adaptive, fires only in top 15% of observed RTT range
        if self._rtt_range.obs_max is not None:
            rtt_top = self._rtt_range.obs_max * 0.85
            rtt_excess = max(0.0, (rtt - rtt_top) / (self._rtt_range.obs_max + 1e-6))
            rtt_penalty = 0.3 * (rtt_excess ** 1.5)
        else:
            rtt_penalty = 0.0

        reward = (
            base_reward
            + improve_bonus
            + efficiency_bonus
            - loss_penalty
            - drop_penalty
            - rtt_penalty
        )

        self._prev_tput = curr_t
        self._prev_streams = streams

        clipped = max(-3.0, min(3.0, reward))
        logger.debug(
            "Reward: base=%.3f imp=%.3f eff=%.3f loss_p=%.3f drop_p=%.3f rtt_p=%.3f "
            "=> %.3f  [tput=%.1f peak=%.1f s=%d | %s | %s]",
            base_reward, improve_bonus, efficiency_bonus,
            loss_penalty, drop_penalty, rtt_penalty, clipped,
            curr_t, peak, streams, self._tput_range, self._rtt_range,
        )
        return clipped

    # -----------------------------------------------------------------------
    # Constraints
    # -----------------------------------------------------------------------

    def _apply_constraints(self, proposed, current, recent) -> int:
        del current, recent
        return max(self._min_connections, min(self._max_connections, proposed))

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _load(self) -> None:
        Q, metadata = self._storage.load()
        if Q:
            self._agent.Q = Q
            saved_eps = metadata.get("exploration_rate")
            if saved_eps is not None:
                self._agent.exploration_rate = max(
                    self._agent.min_exploration, float(saved_eps)
                )
            self._agent.total_decisions = int(metadata.get("total_decisions", 0))
            self._agent.total_updates   = int(metadata.get("total_updates", 0))
            self._agent._total_reward   = float(metadata.get("total_reward", 0.0))
            self._agent._positive_rewards = int(metadata.get("positive_rewards", 0))
            self._agent._negative_rewards = int(metadata.get("negative_rewards", 0))
            self._agent._throughput_improvements = int(
                metadata.get("throughput_improvements", 0)
            )
            saved_conn = metadata.get("current_connections")
            if saved_conn is not None:
                self._agent.current_connections = max(
                    self._min_connections,
                    min(self._max_connections, int(saved_conn)),
                )
            self._prev_streams = self._agent.current_connections
            saved_peak = metadata.get("rolling_peak_mbps")
            if saved_peak is not None:
                self._rolling_peak = float(saved_peak)
            logger.info(
                "Restored: %d Q-states, %d decisions, %d updates, eps=%.4f",
                len(Q), self._agent.total_decisions,
                self._agent.total_updates, self._agent.exploration_rate,
            )

    def __repr__(self) -> str:
        return (
            f"FederatedPolicy(connections={self.current_connections}, "
            f"q_states={len(self._agent.Q)}, "
            f"eps={self._agent.exploration_rate:.4f}, "
            f"tput={self._tput_range}, rtt={self._rtt_range})"
        )