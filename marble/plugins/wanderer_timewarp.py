"""Temporal distortion wanderer plugin."""

from __future__ import annotations

from typing import List, Tuple

import math
import random

from ..reporter import report
from ..wanderer import expose_learnable_params


class TimeWarpPlugin:
    """Occasionally repeat the previous choice based on a learnable strength."""

    @staticmethod
    @expose_learnable_params
    def _params(wanderer, *, warp_strength: float = 0.0):
        return (warp_strength,)

    def choose_next(self, wanderer, current, choices: List[Tuple["Synapse", str]]):
        if not choices:
            return None, "forward"
        (strength_t,) = self._params(wanderer)
        torch = getattr(wanderer, "_torch", None)
        if torch is not None:
            prob = float(torch.sigmoid(strength_t).detach().to("cpu").item())
        else:
            prob = 1.0 / (1.0 + math.exp(-float(strength_t)))
        prev = wanderer._plugin_state.get("timewarp_prev") if hasattr(wanderer, "_plugin_state") else None
        if prev is not None and prev < len(choices) and random.random() < prob:
            idx = int(prev)
        else:
            idx = random.randrange(len(choices))
        wanderer._plugin_state["timewarp_prev"] = idx
        report("wanderer", "timewarp", {"choice": idx}, "plugins")
        return choices[idx]


__all__ = ["TimeWarpPlugin"]

