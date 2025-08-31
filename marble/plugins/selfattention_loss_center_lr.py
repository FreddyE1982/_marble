"""LossCenterLRRoutine steers learning rate toward a target loss."""

from __future__ import annotations

from typing import Any, Dict
import torch

from ..wanderer import expose_learnable_params
from ..reporter import report
from .selfattention_metric_utils import metric_factor


@expose_learnable_params
def _loss_center_params(wanderer, target_loss: float = 0.1, scale: float = 0.5):
    return target_loss, scale


class LossCenterLRRoutine:
    """Adjust ``lr_override`` based on distance from ``target_loss``."""

    def after_step(
        self,
        selfattention: "SelfAttention",
        reporter_ro: Any,
        wanderer: "Wanderer",
        step_index: int,
        ctx: Dict[str, Any],
    ):
        tl_t, sc_t = _loss_center_params(wanderer)
        try:
            target = float(tl_t.detach().to("cpu").item())
        except Exception:
            target = 0.1
        try:
            scale = float(sc_t.detach().to("cpu").item())
        except Exception:
            scale = 0.5
        factor = metric_factor(ctx, "loss_center_lr")
        target *= 1.0 + factor
        scale *= factor
        loss_tensor = ctx.get("cur_loss_tensor")
        if loss_tensor is None:
            return None
        try:
            cur = float(loss_tensor.detach().to("cpu").item())
        except Exception:
            return None
        diff = abs(cur - target)
        lr = float(selfattention.get_param("lr_override") or 0.0)
        selfattention.set_param("lr_override", lr + diff * scale if lr else diff * scale)
        report("selfattention", "loss_center_lr", {"step": step_index, "diff": diff}, "events")
        return None


__all__ = ["LossCenterLRRoutine"]
