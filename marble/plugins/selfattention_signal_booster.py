"""SignalBoosterRoutine amplifies temperature with a learnable gain."""

from __future__ import annotations

from typing import Any, Dict

import torch

from ..wanderer import expose_learnable_params
from ..reporter import report
from .selfattention_metric_utils import metric_factor


@expose_learnable_params
def _boost_param(wanderer, boost_gain: float = 1.0):
    return boost_gain


class SignalBoosterRoutine:
    """Boost attention temperature using a gain parameter."""

    def after_step(self, selfattention: "SelfAttention", reporter_ro: Any, wanderer: "Wanderer", step_index: int, ctx: Dict[str, Any]):
        bg_t = _boost_param(wanderer)
        try:
            gain = float(bg_t.detach().to("cpu").item())
        except Exception:
            gain = 1.0
        mf = metric_factor(ctx, "signal_booster")
        gain *= mf
        try:
            base = float(selfattention.get_param("temperature", 1.0))
            selfattention.set_param("temperature", base * gain)
        except Exception:
            pass
        report(
            "selfattention",
            "signal_booster",
            {"step": step_index, "gain": gain},
            "events",
        )
        return None


__all__ = ["SignalBoosterRoutine"]

