"""WeightDecayRoutine multiplies synapse weights by a decay factor."""

from __future__ import annotations

from typing import Any, Dict
import torch

from ..wanderer import expose_learnable_params
from ..reporter import report


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
            decay = float(d_t.detach().to("cpu").item())
        except Exception:
            decay = 0.01
        for syn in list(getattr(wanderer.brain, "synapses", [])):
            w = getattr(syn, "weight", None)
            if w is None:
                continue
            try:
                wt = torch.tensor([float(w)], dtype=torch.float32)
                syn.weight = float((wt * (1.0 - decay)).detach().to("cpu").item())
            except Exception:
                pass
        report("selfattention", "weight_decay", {"step": step_index, "decay": decay}, "events")
        return None


__all__ = ["WeightDecayRoutine"]
