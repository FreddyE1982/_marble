from __future__ import annotations

"""Helpers for tracking per-plugin execution cost.

This module maintains an exponential moving average (EMA) of the elapsed time
for each plugin.  Call :func:`record` after a plugin finishes executing and use
:func:`get_cost` to retrieve the current average cost.
"""

from typing import Dict

_ALPHA = 0.5
_cost_ema: Dict[str, float] = {}
_ENABLED = False


def enable() -> None:
    """Activate cost profiling."""

    global _ENABLED
    _ENABLED = True


def record(plugin_name: str, elapsed: float) -> None:
    """Record ``elapsed`` time for ``plugin_name``.

    Parameters
    ----------
    plugin_name:
        Name of the plugin whose cost is being tracked.
    elapsed:
        Elapsed execution time in seconds.
    """

    if not _ENABLED:
        return

    val = float(elapsed)
    prev = _cost_ema.get(plugin_name)
    if prev is None:
        _cost_ema[plugin_name] = val
    else:
        _cost_ema[plugin_name] = _ALPHA * val + (1.0 - _ALPHA) * prev


def get_cost(plugin_name: str, default: float = float("nan")) -> float:
    """Return the current cost estimate for ``plugin_name``.

    Parameters
    ----------
    plugin_name:
        Name of the plugin.
    default:
        Value returned if no cost has been recorded yet.  Defaults to ``NaN`` so
        callers can detect whether the cost is unknown and fall back to
        alternative logic.
    """

    return float(_cost_ema.get(plugin_name, default))


__all__ = ["record", "get_cost", "enable"]
