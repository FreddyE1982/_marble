"""Utility to decide plugin actions with constraints.

This controller selects plugin actions while respecting incompatibility sets,
per-plugin capacity limits and a global budget read from ``config.yaml``.

The function :func:`decide_actions` accepts the current plugin hints ``h_t``,
proposed actions ``x_t`` and a ``history`` of previous action sets. It returns a
subset of actions that satisfy all constraints.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Iterable, List, Any, Set

import torch

from .constraints import check_budget, check_incompatibility, check_throughput
from .plugin_graph import PLUGIN_GRAPH

# Incompatibility sets I_t: mapping plugin name to set of incompatible plugins
INCOMPATIBILITY_SETS: Dict[str, Set[str]] = {
    "A": {"C"},
    "C": {"A"},
}

# Capacity limits c: maximum times a plugin can appear in history + current step
CAPACITY_LIMITS: Dict[str, int] = {
    "A": 2,
    "B": 1,
    "C": 1,
}


def _load_budget() -> float:
    """Load budget limit from ``config.yaml``."""
    base = os.path.dirname(os.path.dirname(__file__))
    cfg: Dict[str, Dict[str, Any]] = {}
    try:
        with open(os.path.join(base, "config.yaml"), "r", encoding="utf-8") as fh:
            section: str | None = None
            for raw in fh:
                line = raw.split("#", 1)[0].rstrip()
                if not line:
                    continue
                if not line.startswith(" ") and line.endswith(":"):
                    section = line[:-1].strip()
                    cfg[section] = {}
                    continue
                if section and ":" in line:
                    k, v = line.split(":", 1)
                    try:
                        cfg[section][k.strip()] = float(v.strip())
                    except Exception:
                        cfg[section][k.strip()] = v.strip()
    except Exception:
        return float("inf")
    dc = cfg.get("decision_controller", {})
    try:
        return float(dc.get("budget", float("inf")))
    except Exception:
        return float("inf")


BUDGET_LIMIT = _load_budget()


def _load_l1_penalty() -> float:
    """Load L1 penalty for contribution regressor from ``config.yaml``."""
    base = os.path.dirname(os.path.dirname(__file__))
    cfg: Dict[str, Dict[str, Any]] = {}
    try:
        with open(os.path.join(base, "config.yaml"), "r", encoding="utf-8") as fh:
            section: str | None = None
            for raw in fh:
                line = raw.split("#", 1)[0].rstrip()
                if not line:
                    continue
                if not line.startswith(" ") and line.endswith(":"):
                    section = line[:-1].strip()
                    cfg[section] = {}
                    continue
                if section and ":" in line:
                    k, v = line.split(":", 1)
                    try:
                        cfg[section][k.strip()] = float(v.strip())
                    except Exception:
                        cfg[section][k.strip()] = v.strip()
    except Exception:
        return 0.0
    dc = cfg.get("decision_controller", {})
    try:
        return float(dc.get("contribution_l1", 0.0))
    except Exception:
        return 0.0


L1_PENALTY = _load_l1_penalty()


def _load_tau_threshold() -> float:
    """Load minimum state-change interval from ``config.yaml``."""
    base = os.path.dirname(os.path.dirname(__file__))
    cfg: Dict[str, Dict[str, Any]] = {}
    try:
        with open(os.path.join(base, "config.yaml"), "r", encoding="utf-8") as fh:
            section: str | None = None
            for raw in fh:
                line = raw.split("#", 1)[0].rstrip()
                if not line:
                    continue
                if not line.startswith(" ") and line.endswith(":"):
                    section = line[:-1].strip()
                    cfg[section] = {}
                    continue
                if section and ":" in line:
                    k, v = line.split(":", 1)
                    try:
                        cfg[section][k.strip()] = float(v.strip())
                    except Exception:
                        cfg[section][k.strip()] = v.strip()
    except Exception:
        return 0.0
    dc = cfg.get("decision_controller", {})
    try:
        return float(dc.get("tau_threshold", 0.0))
    except Exception:
        return 0.0


TAU_THRESHOLD = _load_tau_threshold()

# Track last state-change timestamp for each plugin
LAST_STATE_CHANGE: Dict[str, float] = {}


def record_plugin_state_change(name: str, now: float | None = None) -> None:
    """Record that ``name`` changed state at time ``now``."""
    if now is None:
        now = time.time()
    LAST_STATE_CHANGE[name] = float(now)


def tau_since_last_change(name: str, now: float | None = None) -> float:
    """Return seconds since ``name`` last changed state."""
    if now is None:
        now = time.time()
    last = LAST_STATE_CHANGE.get(name)
    if last is None:
        return float("inf")
    return float(now - last)


def train_contribution_regressor(
    activation: torch.Tensor,
    outcomes: torch.Tensor,
    l1_penalty: float | None = None,
    lr: float = 0.01,
    epochs: int = 1000,
) -> torch.Tensor:
    """Train linear regressor ``q_a(z)`` with ℓ1 penalty.

    Parameters
    ----------
    activation:
        Matrix of plugin activations with shape ``(samples, plugins)``.
    outcomes:
        Target outcomes corresponding to each row in ``activation``.
    l1_penalty:
        Strength of the ℓ1 penalty. When ``None``, uses value from config.
    lr:
        Learning rate for simple gradient descent.
    epochs:
        Number of optimization steps.

    Returns
    -------
    torch.Tensor
        Learned weight vector of shape ``(plugins,)``.
    """

    if l1_penalty is None:
        l1_penalty = L1_PENALTY
    weights = torch.zeros(activation.shape[1], device=activation.device, requires_grad=True)
    for _ in range(epochs):
        pred = activation @ weights
        loss = ((pred - outcomes) ** 2).mean() + l1_penalty * weights.abs().sum()
        loss.backward()
        with torch.no_grad():
            weights -= lr * weights.grad
            weights.grad.zero_()
    return weights.detach()


def estimate_plugin_contributions(
    activation: torch.Tensor,
    outcomes: torch.Tensor,
    plugin_names: List[str],
    l1_penalty: float | None = None,
) -> Dict[str, float]:
    """Estimate per-plugin contribution scores.

    The contribution score for each plugin corresponds to the learned weight
    of the ℓ1-regularized regressor.
    """

    weights = train_contribution_regressor(activation, outcomes, l1_penalty)
    return {
        name: float(w.detach().to("cpu").item()) for name, w in zip(plugin_names, weights)
    }


def decide_actions(
    h_t: Dict[str, Dict[str, float]],
    x_t: Dict[str, Any],
    history: Iterable[Dict[str, Any]],
    contrib_scores: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Select actions while enforcing incompatibilities, capacity and budget.

    Parameters
    ----------
    h_t:
        Mapping of plugin names to metadata containing at least a ``cost`` key.
    x_t:
        Proposed actions for plugins, typically produced by a planner.
    history:
        Iterable of previous action dictionaries. Used to enforce capacity
        limits across time.
    contrib_scores:
        Optional mapping of plugin names to contribution scores. Higher scores
        effectively reduce the plugin's cost during selection.

    Returns
    -------
    Dict[str, Any]
        Selected subset of ``x_t`` satisfying all constraints.
    """

    usage: Dict[str, int] = {}
    running_costs: Dict[str, float] = {}
    for past in history:
        for name in past:
            usage[name] = usage.get(name, 0) + 1
            cost = float(h_t.get(name, {}).get("cost", 0.0))
            running_costs[name] = running_costs.get(name, 0.0) + cost

    now = time.time()

    def penalty(name: str) -> float:
        tau = tau_since_last_change(name, now)
        if tau < TAU_THRESHOLD:
            return TAU_THRESHOLD - tau
        return 0.0

    # Sort candidates by effective cost including penalty so recent state changes
    # are deprioritized under the budget constraint
    ordered = sorted(
        x_t.items(),
        key=lambda kv: h_t.get(kv[0], {}).get("cost", 0.0)
        + penalty(kv[0])
        - (contrib_scores.get(kv[0], 0.0) if contrib_scores else 0.0),
    )

    selected: Dict[str, Any] = {}
    active: Set[str] = set()
    remaining = BUDGET_LIMIT

    for name, action in ordered:
        base_cost = float(h_t.get(name, {}).get("cost", 0.0))
        pen = penalty(name)
        cost = base_cost + pen
        if contrib_scores:
            cost -= float(contrib_scores.get(name, 0.0))
        if not check_throughput(name, usage, CAPACITY_LIMITS):
            continue
        if not check_budget(name, cost, remaining, running_costs, BUDGET_LIMIT):
            continue
        if not check_incompatibility(name, active, INCOMPATIBILITY_SETS):
            continue
        selected[name] = action
        active.add(name)
        usage[name] = usage.get(name, 0) + 1
        running_costs[name] = running_costs.get(name, 0.0) + cost
        remaining -= cost
        record_plugin_state_change(name, now)
        PLUGIN_GRAPH.mark_executed(name)
        if remaining <= 0:
            break
    return selected


__all__ = [
    "decide_actions",
    "INCOMPATIBILITY_SETS",
    "CAPACITY_LIMITS",
    "BUDGET_LIMIT",
    "train_contribution_regressor",
    "estimate_plugin_contributions",
    "L1_PENALTY",
    "TAU_THRESHOLD",
    "LAST_STATE_CHANGE",
    "record_plugin_state_change",
    "tau_since_last_change",
]
