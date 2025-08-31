"""StepFaderRoutine gradually lowers temperature with every step."""

from __future__ import annotations

from typing import Any, Dict
import torch

from ..wanderer import expose_learnable_params
from ..reporter import report
from .selfattention_metric_utils import metric_factor


@expose_learnable_params
def _step_fader_params(wanderer, slope: float = 0.01):
    return slope


class StepFaderRoutine:
    """Linearly decrease temperature by ``slope`` each step."""

    def after_step(
        self,
        selfattention: "SelfAttention",
        reporter_ro: Any,
        wanderer: "Wanderer",
        step_index: int,
        ctx: Dict[str, Any],
    ):
        s_t = _step_fader_params(wanderer)
        try:
            slope = float(s_t.detach().to("cpu").item())
        except Exception:
            slope = 0.01
        slope *= metric_factor(ctx, "step_fader")
        try:
            base = float(selfattention.get_param("temperature", 1.0))
            new_temp = max(0.0, base - slope * float(step_index))
            selfattention.set_param("temperature", new_temp)
        except Exception:
            new_temp = float("nan")
        report("selfattention", "step_fader", {"step": step_index, "temp": new_temp}, "events")
        return None


__all__ = ["StepFaderRoutine"]
