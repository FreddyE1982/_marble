"""Utility to decide plugin actions with constraints.

This controller selects plugin actions while respecting incompatibility sets,
per-plugin capacity limits and a global budget read from ``config.yaml``.

The function :func:`decide_actions` accepts the current plugin hints ``h_t``,
proposed actions ``x_t`` and a ``history`` of previous action sets. It returns a
subset of actions that satisfy all constraints.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Any, Set

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


def decide_actions(h_t: Dict[str, Dict[str, float]], x_t: Dict[str, Any], history: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
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

    Returns
    -------
    Dict[str, Any]
        Selected subset of ``x_t`` satisfying all constraints.
    """

    usage: Dict[str, int] = {}
    for past in history:
        for name in past:
            usage[name] = usage.get(name, 0) + 1

    # Sort candidates by cost so cheaper actions are preferred under budget
    ordered = sorted(x_t.items(), key=lambda kv: h_t.get(kv[0], {}).get("cost", 0.0))

    selected: Dict[str, Any] = {}
    active: Set[str] = set()
    remaining = BUDGET_LIMIT

    for name, action in ordered:
        cost = float(h_t.get(name, {}).get("cost", 0.0))
        cap = CAPACITY_LIMITS.get(name, float("inf"))
        if usage.get(name, 0) >= cap:
            continue
        if cost > remaining:
            continue
        incompatible = INCOMPATIBILITY_SETS.get(name, set())
        if active & incompatible:
            continue
        selected[name] = action
        active.add(name)
        usage[name] = usage.get(name, 0) + 1
        remaining -= cost
        if remaining <= 0:
            break
    return selected


__all__ = ["decide_actions", "INCOMPATIBILITY_SETS", "CAPACITY_LIMITS", "BUDGET_LIMIT"]
