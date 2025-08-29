from __future__ import annotations

"""Chaotic logistic-map driven wanderer plugin."""

import math
import torch
from typing import List, Tuple

from ..wanderer import expose_learnable_params


class ChaosWalkPlugin:
    """Selects next synapse via logistic-map chaos sequence."""

    def __init__(self) -> None:
        self._state = torch.rand(1)

    @staticmethod
    @expose_learnable_params
    def _params(wanderer, *, chaos_lambda: float = 3.7):
        return (chaos_lambda,)

    def choose_next(self, wanderer, current, choices: List[Tuple["Synapse", str]]):
        if not choices:
            return None, "forward"
        (lam_t,) = self._params(wanderer)
        lam = float(lam_t.detach().to("cpu").item()) if hasattr(lam_t, "detach") else float(lam_t)
        self._state = lam * self._state * (1 - self._state)
        idx = int(self._state.item() * len(choices)) % len(choices)
        return choices[idx]


__all__ = ["ChaosWalkPlugin"]

PLUGIN_NAME = "chaoswalk"

