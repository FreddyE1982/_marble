"""Shadow clone wanderer plugin."""

from __future__ import annotations

from typing import List, Tuple

import math
import random

from ..reporter import report
from ..wanderer import expose_learnable_params


class ShadowClonePlugin:
    """Randomly flip direction creating a 'shadow' path."""

    @staticmethod
    @expose_learnable_params
    def _params(wanderer, *, clone_bias: float = 0.0):
        return (clone_bias,)

    def choose_next(self, wanderer, current, choices: List[Tuple["Synapse", str]]):
        if not choices:
            return None, "forward"
        (bias_t,) = self._params(wanderer)
        torch = getattr(wanderer, "_torch", None)
        if torch is not None:
            prob = float(torch.sigmoid(bias_t).detach().to(wanderer._device).item())
        else:
            prob = 1.0 / (1.0 + math.exp(-float(bias_t)))
        idx = random.randrange(len(choices))
        syn, direction = choices[idx]
        if random.random() < prob:
            direction = "backward" if direction == "forward" else "forward"
        report("wanderer", "shadow_clone", {"choice": idx, "dir": direction}, "plugins")
        return syn, direction


__all__ = ["ShadowClonePlugin"]

