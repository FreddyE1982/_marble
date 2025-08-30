"""Wanderer plugin that prefers higher-dimensional targets."""

from __future__ import annotations

from typing import List, Tuple

import math
import random

from ..reporter import report
from ..wanderer import expose_learnable_params


class DimensionalShiftPlugin:
    """Bias selection toward synapses whose target has many coordinates.

    A single learnable bias term influences the softmax weighting so the
    wanderer can learn whether to favour or avoid high-dimensional regions.
    """

    @staticmethod
    @expose_learnable_params
    def _params(wanderer, *, shift_bias: float = 0.0):
        return (shift_bias,)

    def choose_next(self, wanderer, current, choices: List[Tuple["Synapse", str]]):
        if not choices:
            return None, "forward"
        (bias_t,) = self._params(wanderer)
        torch = getattr(wanderer, "_torch", None)
        device = getattr(wanderer, "_device", "cpu")
        dims: List[float] = []
        for syn, direction in choices:
            target = syn.target if direction == "forward" else syn.source
            pos = getattr(target, "position", ()) or ()
            dims.append(float(len(pos)))
        if torch is not None:
            t_dims = torch.tensor(dims, dtype=torch.float32, device=device)
            weights = torch.softmax(t_dims + bias_t, dim=0)
            idx = int(torch.multinomial(weights, 1).item())
        else:
            exps = [math.exp(d + float(bias_t)) for d in dims]
            total = sum(exps)
            r = random.random() * total
            upto = 0.0
            idx = 0
            for i, e in enumerate(exps):
                upto += e
                if upto >= r:
                    idx = i
                    break
        report("wanderer", "dimensional_shift", {"choice": idx}, "plugins")
        return choices[idx]


__all__ = ["DimensionalShiftPlugin"]

