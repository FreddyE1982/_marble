from __future__ import annotations

"""Fractal seed wanderer plugin."""

import torch
from typing import List, Tuple

from ..wanderer import expose_learnable_params


class FractalSeedPlugin:
    """Uses fractional components of a sequence to guide exploration."""

    def __init__(self) -> None:
        self._counter = 0

    @staticmethod
    @expose_learnable_params
    def _params(wanderer, *, fractal_depth: float = 2.0):
        return (fractal_depth,)

    def choose_next(self, wanderer, current, choices: List[Tuple["Synapse", str]]):
        if not choices:
            return None, "forward"
        (d_t,) = self._params(wanderer)
        d = float(d_t.detach().to("cpu").item()) if hasattr(d_t, "detach") else float(d_t)
        base = torch.arange(len(choices), dtype=torch.float32)
        self._counter += 1
        weights = torch.frac(base * d * (self._counter)) + 1e-6
        probs = weights / weights.sum()
        idx = int(torch.multinomial(probs, 1).item())
        return choices[idx]


__all__ = ["FractalSeedPlugin"]

PLUGIN_NAME = "fractalseed"

