"""TimeGateTemperatureRoutine applies a gated temperature schedule."""

from __future__ import annotations

from typing import Any, Dict
import torch

from ..wanderer import expose_learnable_params
from ..reporter import report
from .selfattention_metric_utils import metric_factor


@expose_learnable_params
def _time_gate_params(wanderer, interval: float = 5.0, gated_temp: float = 0.5):
    return interval, gated_temp


class TimeGateTemperatureRoutine:
    """Set temperature to ``gated_temp`` every ``interval`` steps."""

    def after_step(
        self,
        selfattention: "SelfAttention",
        reporter_ro: Any,
        wanderer: "Wanderer",
        step_index: int,
        ctx: Dict[str, Any],
    ):
        it_t, gt_t = _time_gate_params(wanderer)
        try:
            interval = max(1, int(float(it_t.detach().to("cpu").item())))
        except Exception:
            interval = 5
        try:
            gated_temp = float(gt_t.detach().to("cpu").item())
        except Exception:
            gated_temp = 0.5
        mf = metric_factor(ctx, "time_gate_temp")
        interval = max(1, int(interval * (1.0 + mf)))
        gated_temp *= mf
        if step_index % interval == 0:
            try:
                selfattention.set_param("temperature", gated_temp)
            except Exception:
                pass
            report("selfattention", "time_gate_temp", {"step": step_index, "temp": gated_temp}, "events")
        return None


__all__ = ["TimeGateTemperatureRoutine"]
