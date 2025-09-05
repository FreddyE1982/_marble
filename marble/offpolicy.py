"""Off-policy evaluation utilities.

This module provides helpers to log historical actions and rewards,
compute importance weights between a logging policy and a new policy,
and estimate policy value via the doubly-robust estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence


@dataclass
class Trajectory:
    """Container for logged interaction data.

    Attributes
    ----------
    actions:
        Sequence of action identifiers taken by the logging policy.
    rewards:
        Observed rewards for each step.
    logged_probs:
        Probabilities of actions under the logging policy.
    new_probs:
        Probabilities of the same actions under the new policy.
    """

    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    logged_probs: List[float] = field(default_factory=list)
    new_probs: List[float] = field(default_factory=list)

    def log(self, action: int, reward: float, logged_prob: float, new_prob: float) -> None:
        """Append a new step to the trajectory."""
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.logged_probs.append(float(logged_prob))
        self.new_probs.append(float(new_prob))


# ---------------------------------------------------------------------------
# Importance sampling weights
# ---------------------------------------------------------------------------

def importance_weights(logged: Sequence[float], new: Sequence[float]) -> List[float]:
    """Return cumulative importance weights ``ρ_t``.

    ``ρ_t`` is the product of ratios up to step ``t`` of the new policy
    probability over the logging policy probability.
    """

    ratios: List[float] = []
    cum = 1.0
    for lp, np in zip(logged, new):
        if lp <= 0:
            cum = 0.0
        else:
            cum *= np / lp
        ratios.append(cum)
    return ratios


# ---------------------------------------------------------------------------
# Doubly-robust estimator
# ---------------------------------------------------------------------------

def doubly_robust(traj: Trajectory, q_hat: Sequence[float]) -> float:
    """Return doubly-robust value estimate ``V_hat``.

    This implementation follows a backward recursion where the baseline
    value for the next step is multiplied by the per-step importance ratio
    before being carried backward.  As a result, identical policies with
    zero rewards leave the initial baseline unchanged.

    Parameters
    ----------
    traj:
        Logged trajectory with probabilities for logging and target policies.
    q_hat:
        Baseline action-value estimates ``Q_hat`` for each step (including
        a bootstrap value for the terminal state).  Its length must be
        ``len(traj.actions) + 1``.
    """

    if len(q_hat) != len(traj.actions) + 1:
        raise ValueError("q_hat must have one more element than actions")

    v_hat = q_hat[-1]
    for t in reversed(range(len(traj.rewards))):
        lp = traj.logged_probs[t]
        np = traj.new_probs[t]
        w_t = np / lp if lp > 0 else 0.0
        v_hat = q_hat[t] + w_t * (traj.rewards[t] + v_hat - q_hat[t + 1])
    return float(v_hat)


__all__ = ["Trajectory", "importance_weights", "doubly_robust"]
