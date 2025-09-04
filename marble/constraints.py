from __future__ import annotations

"""Utility checks for plugin selection constraints.

This module exposes helpers used by :mod:`marble.decision_controller` to
validate proposed plugin actions.  The functions are intentionally side-effect
free so callers can compose them flexibly while tracking any required state.
"""

from typing import Dict, Set


def check_budget(name: str, cost: float, remaining: float, running_costs: Dict[str, float], budget_limit: float) -> bool:
    """Return ``True`` if a plugin respects budget constraints.

    Parameters
    ----------
    name:
        Plugin identifier.
    cost:
        Cost of the proposed action for ``name``.
    remaining:
        Remaining global budget available for this decision step.
    running_costs:
        Mapping of plugin names to their accumulated running costs ``b_i``.
    budget_limit:
        Global budget limit ``B``.  Each plugin's running cost must remain
        below this threshold.
    """
    if cost > remaining:
        return False
    return running_costs.get(name, 0.0) + cost <= budget_limit


def check_incompatibility(name: str, active: Set[str], incompatibilities: Dict[str, Set[str]]) -> bool:
    """Return ``True`` if ``name`` is not incompatible with active plugins."""
    incompatible = incompatibilities.get(name, set())
    return not bool(active & incompatible)


def check_throughput(name: str, usage: Dict[str, int], limits: Dict[str, int]) -> bool:
    """Return ``True`` if plugin throughput does not exceed its limit."""
    return usage.get(name, 0) < limits.get(name, float("inf"))


__all__ = ["check_budget", "check_incompatibility", "check_throughput"]
