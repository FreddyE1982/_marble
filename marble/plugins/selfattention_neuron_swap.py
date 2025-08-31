"""NeuronSwapRoutine swaps tensors of random neuron pairs."""

from __future__ import annotations

from typing import Any, Dict
import torch

from ..wanderer import expose_learnable_params
from ..reporter import report
from .selfattention_metric_utils import metric_factor


@expose_learnable_params
def _swap_param(wanderer, swap_prob: float = 0.1):
    return swap_prob


class NeuronSwapRoutine:
    """Randomly swap neuron tensors with probability ``swap_prob``."""

    def after_step(
        self,
        selfattention: "SelfAttention",
        reporter_ro: Any,
        wanderer: "Wanderer",
        step_index: int,
        ctx: Dict[str, Any],
    ):
        sp_t = _swap_param(wanderer)
        try:
            prob = float(sp_t.detach().to("cpu").item())
        except Exception:
            prob = 0.1
        prob *= metric_factor(ctx, "neuron_swap")
        neurons = list(getattr(wanderer.brain, "neurons", {}).values())
        if len(neurons) >= 2 and torch.rand(1).item() < prob:
            idxs = torch.randint(0, len(neurons), (2,))
            a, b = neurons[int(idxs[0])], neurons[int(idxs[1])]
            a.tensor, b.tensor = b.tensor, a.tensor
            report("selfattention", "neuron_swap", {"step": step_index}, "events")
        return None


__all__ = ["NeuronSwapRoutine"]
