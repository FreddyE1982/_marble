"""WeightDecayRoutine multiplies synapse weights by a decay factor."""

from __future__ import annotations

from typing import Any, Dict
import torch

from ..wanderer import expose_learnable_params
from ..reporter import report
from .selfattention_metric_utils import metric_factor


@expose_learnable_params
def _decay_param(wanderer, decay: float = 0.01):
    return decay


class WeightDecayRoutine:
    """Apply ``decay`` to all synapse weights each step."""

    def after_step(
        self,
        selfattention: "SelfAttention",
        reporter_ro: Any,
        wanderer: "Wanderer",
        step_index: int,
        ctx: Dict[str, Any],
    ):
        d_t = _decay_param(wanderer)
        try:
            decay = float(d_t.detach().to(wanderer._device).item())
        except Exception:
            decay = 0.01
        decay *= metric_factor(ctx, "weight_decay")
        for syn in list(getattr(wanderer.brain, "synapses", [])):
            w = getattr(syn, "weight", None)
            if w is None:
                continue
            try:
                wt = torch.tensor([float(w)], dtype=torch.float32, device=wanderer._device)
                syn.weight = float((wt * (1.0 - decay)).detach().to(wanderer._device).item())
            except Exception:
                pass
        report("selfattention", "weight_decay", {"step": step_index, "decay": decay}, "events")
        return None


__all__ = ["WeightDecayRoutine"]
