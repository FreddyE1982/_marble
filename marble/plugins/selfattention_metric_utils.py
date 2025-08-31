"""Utility helpers for SelfAttention metric-driven plugins."""

from __future__ import annotations

from typing import Any, Dict

from ..reporter import report


def metric_factor(ctx: Dict[str, Any], tag: str) -> float:
    """Extract standard self-attention metrics and log their usage.

    Parameters
    ----------
    ctx: Dict[str, Any]
        Context passed to ``after_step`` containing metric values.
    tag: str
        Identifier for the calling plugin so metrics can be traced in the
        reporter output.

    Returns
    -------
    float
        A scaling factor ``1 / (1 + |loss| + |speed| + |accel| + complexity)``
        that plugins can use to modulate their behaviour.
    """

    loss = float(ctx.get("sa_loss", 0.0) or 0.0)
    speed = float(ctx.get("sa_loss_speed", 0.0) or 0.0)
    accel = float(ctx.get("sa_loss_accel", 0.0) or 0.0)
    complexity = float(ctx.get("sa_model_complexity", 0.0) or 0.0)
    try:
        report(
            "selfattention",
            f"{tag}_metrics",
            {
                "loss": loss,
                "speed": speed,
                "accel": accel,
                "complexity": complexity,
            },
            "metrics",
        )
    except Exception:
        pass
    return 1.0 / (1.0 + abs(loss) + abs(speed) + abs(accel) + complexity)


__all__ = ["metric_factor"]

