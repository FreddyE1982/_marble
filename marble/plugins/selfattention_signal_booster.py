"""SignalBoosterRoutine amplifies temperature with a learnable gain."""

from __future__ import annotations

from typing import Any, Dict

import torch

from ..wanderer import expose_learnable_params
from ..reporter import report


@expose_learnable_params
def _boost_param(wanderer, boost_gain: float = 1.0):
    return boost_gain


class SignalBoosterRoutine:
    """Boost attention temperature using a gain parameter."""

    def after_step(self, selfattention: "SelfAttention", reporter_ro: Any, wanderer: "Wanderer", step_index: int, ctx: Dict[str, Any]):
        bg_t = _boost_param(wanderer)
        try:
            gain_t = torch.tensor(bg_t.detach().to("cpu").item(), dtype=torch.float32)
        except Exception:
            gain_t = torch.tensor(1.0)
        loss = float(ctx.get("sa_loss", 0.0) or 0.0)
        speed = float(ctx.get("sa_loss_speed", 0.0) or 0.0)
        accel = float(ctx.get("sa_loss_accel", 0.0) or 0.0)
        complexity = float(ctx.get("sa_model_complexity", 0.0) or 0.0)
        gain_t = gain_t / (1.0 + torch.abs(torch.tensor(loss)))
        gain_t = gain_t / (1.0 + torch.abs(torch.tensor(speed)))
        gain_t = gain_t / (1.0 + torch.abs(torch.tensor(accel)))
        gain_t = gain_t / (1.0 + torch.tensor(complexity))
        gain = float(gain_t.detach().to("cpu").item())
        try:
            base = float(selfattention.get_param("temperature", 1.0))
            selfattention.set_param("temperature", base * gain)
        except Exception:
            pass
        report(
            "selfattention",
            "signal_booster",
            {
                "step": step_index,
                "gain": gain,
                "loss": loss,
                "speed": speed,
                "accel": accel,
                "complexity": complexity,
            },
            "events",
        )
        return None


__all__ = ["SignalBoosterRoutine"]

