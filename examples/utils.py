"""Utility helpers for example scripts.

This module exposes convenience functions shared by example scripts.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

__all__ = ["decide_with_pred"]


def decide_with_pred(
    dc: Any,
    h_t: Dict[str, Dict[str, float]],
    pred: Any,
    target: Optional[Any] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Invoke a decision controller using a prediction tensor.

    Parameters
    ----------
    dc:
        Decision controller instance providing a :meth:`decide` method.
    h_t:
        Mapping of plugin names to per-step hint dictionaries.
    pred:
        Prediction tensor used to build the context sequence.
    target:
        Optional target tensor concatenated with ``pred`` to form the context.
    metrics:
        Optional reporter metrics passed through to :meth:`dc.decide`.

    Returns
    -------
    Dict[str, Any]
        The decision information returned by the controller.

    Examples
    --------
    >>> # Assuming ``controller`` is a DecisionController
    >>> hints = {"plugin": {"cost": 1.0}}
    >>> decide_with_pred(controller, hints, pred_tensor)
    {{'plugin': 'selected'}}
    """
    return dc.decide(h_t, pred=pred, target=target, metrics=metrics)
