from __future__ import annotations

"""Helpers for tracking per-plugin execution cost.

This module maintains an exponential moving average (EMA) of the elapsed time
for each plugin.  Call :func:`record` after a plugin finishes executing and use
:func:`get_cost` to retrieve the current average cost.
"""

from typing import Dict

_ALPHA = 0.5
_cost_ema: Dict[str, float] = {}


def record(plugin_name: str, elapsed: float) -> None:
    """Record ``elapsed`` time for ``plugin_name``.

    Parameters
    ----------
    plugin_name:
        Name of the plugin whose cost is being tracked.
    elapsed:
        Elapsed execution time in seconds.
    """

    val = float(elapsed)
    prev = _cost_ema.get(plugin_name)
    if prev is None:
        _cost_ema[plugin_name] = val
    else:
        _cost_ema[plugin_name] = _ALPHA * val + (1.0 - _ALPHA) * prev


def get_cost(plugin_name: str, default: float = 0.0) -> float:
    """Return the current cost estimate for ``plugin_name``.

    Parameters
    ----------
    plugin_name:
        Name of the plugin.
    default:
        Value returned if no cost has been recorded yet.
    """

    return float(_cost_ema.get(plugin_name, default))


__all__ = ["record", "get_cost"]
