"""SynapseNoiseRoutine injects Gaussian noise into synapse weights."""

from __future__ import annotations

from typing import Any, Dict
import torch

from ..wanderer import expose_learnable_params
from ..reporter import report


@expose_learnable_params
def _noise_param(wanderer, noise_std: float = 0.01):
    return noise_std


class SynapseNoiseRoutine:
    """Add random noise with std ``noise_std`` to synapse weights."""

    def after_step(
        self,
        selfattention: "SelfAttention",
        reporter_ro: Any,
        wanderer: "Wanderer",
        step_index: int,
        ctx: Dict[str, Any],
    ):
        ns_t = _noise_param(wanderer)
        try:
            std = float(ns_t.detach().to("cpu").item())
        except Exception:
            std = 0.01
        noise = torch.randn(1).item() * std
        for syn in list(getattr(wanderer.brain, "synapses", [])):
            w = getattr(syn, "weight", None)
            if w is None:
                continue
            syn.weight = float(torch.tensor([float(w) + noise]).detach().to("cpu").item())
        report("selfattention", "synapse_noise", {"step": step_index, "std": std}, "events")
        return None


__all__ = ["SynapseNoiseRoutine"]
