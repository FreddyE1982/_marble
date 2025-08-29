from __future__ import annotations

"""Quantum-inspired superposition wanderer plugin."""

import random
import torch
from typing import List, Tuple

from ..wanderer import expose_learnable_params


class QuantumSuperpositionPlugin:
    """Collapses to random choice with learnable probability."""

    @staticmethod
    @expose_learnable_params
    def _params(wanderer, *, collapse_prob: float = 0.1):
        return (collapse_prob,)

    def choose_next(self, wanderer, current, choices: List[Tuple["Synapse", str]]):
        if not choices:
            return None, "forward"
        (p_t,) = self._params(wanderer)
        p = float(p_t.detach().to("cpu").item()) if hasattr(p_t, "detach") else float(p_t)
        if random.random() < p:
            return random.choice(choices)
        weights = torch.arange(1, len(choices) + 1, dtype=torch.float32)
        probs = weights / weights.sum()
        idx = int(torch.multinomial(probs, 1).item())
        return choices[idx]


__all__ = ["QuantumSuperpositionPlugin"]

PLUGIN_NAME = "quantumsuper"

