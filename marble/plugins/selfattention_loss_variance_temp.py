"""LossVarianceTemperatureRoutine scales temperature by recent loss variance."""

from __future__ import annotations

from typing import Any, Dict
import torch

from ..wanderer import expose_learnable_params
from ..reporter import report
from .selfattention_metric_utils import metric_factor


@expose_learnable_params
def _loss_var_params(wanderer, window: float = 5.0, factor: float = 0.1):
    return window, factor


class LossVarianceTemperatureRoutine:
    """Adjust temperature using variance of recent losses."""

    def after_step(
        self,
        selfattention: "SelfAttention",
        reporter_ro: Any,
        wanderer: "Wanderer",
        step_index: int,
        ctx: Dict[str, Any],
    ):
        w_t, f_t = _loss_var_params(wanderer)
        try:
            window = max(1, int(float(w_t.detach().to("cpu").item())))
        except Exception:
            window = 5
        try:
            factor = float(f_t.detach().to("cpu").item())
        except Exception:
            factor = 0.1
        mfac = metric_factor(ctx, "loss_variance_temp")
        window = max(1, int(window * (1.0 + mfac)))
        factor *= mfac
        losses = []
        for i in range(max(0, step_index - window), step_index):
            try:
                item = reporter_ro.item(f"step_{i}", "wanderer_steps", "logs")
                if isinstance(item, dict) and ("current_loss" in item):
                    losses.append(float(item["current_loss"]))
            except Exception:
                pass
        if losses:
            var = float(torch.var(torch.tensor(losses)).detach().to("cpu").item())
            try:
                base = float(selfattention.get_param("temperature", 1.0))
                selfattention.set_param("temperature", base * (1.0 + factor * var))
            except Exception:
                pass
            report("selfattention", "loss_variance_temp", {"step": step_index, "var": var}, "events")
        return None


__all__ = ["LossVarianceTemperatureRoutine"]
