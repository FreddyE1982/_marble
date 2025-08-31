"""SynapseRenormRoutine enforces a target weight norm on synapses."""

from __future__ import annotations

from typing import Any, Dict
import torch

from ..wanderer import expose_learnable_params
from ..reporter import report
from .selfattention_metric_utils import metric_factor


@expose_learnable_params
def _renorm_param(wanderer, target_norm: float = 1.0):
    return target_norm


class SynapseRenormRoutine:
    """Renormalize synapse weights to ``target_norm`` L2 norm."""

    def after_step(
        self,
        selfattention: "SelfAttention",
        reporter_ro: Any,
        wanderer: "Wanderer",
        step_index: int,
        ctx: Dict[str, Any],
    ):
        tn_t = _renorm_param(wanderer)
        try:
            target = float(tn_t.detach().to("cpu").item())
        except Exception:
            target = 1.0
        target *= metric_factor(ctx, "synapse_renorm")
        for syn in list(getattr(wanderer.brain, "synapses", [])):
            w = getattr(syn, "weight", None)
            if w is None:
                continue
            try:
                wt = torch.tensor([float(w)], dtype=torch.float32)
                norm = torch.norm(wt)
                if norm > 0:
                    syn.weight = float((wt / norm * target).detach().to("cpu").item())
            except Exception:
                pass
        report("selfattention", "synapse_renorm", {"step": step_index}, "events")
        return None


__all__ = ["SynapseRenormRoutine"]
