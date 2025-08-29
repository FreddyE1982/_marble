from __future__ import annotations

"""Memory decay wanderer plugin."""

import math
import torch
from typing import Dict, List, Tuple

from ..wanderer import expose_learnable_params


class MemoryDecayPlugin:
    """Avoids recently visited nodes with exponential decay memory."""

    def __init__(self) -> None:
        self._memory: Dict[object, float] = {}

    @staticmethod
    @expose_learnable_params
    def _params(wanderer, *, memory_half_life: float = 10.0):
        return (memory_half_life,)

    def choose_next(self, wanderer, current, choices: List[Tuple["Synapse", str]]):
        if not choices:
            return None, "forward"
        (hl_t,) = self._params(wanderer)
        hl = float(hl_t.detach().to("cpu").item()) if hasattr(hl_t, "detach") else float(hl_t)
        decay = math.log(2.0) / max(hl, 1e-6)
        for k in list(self._memory.keys()):
            self._memory[k] *= math.exp(-decay)
        self._memory[current] = self._memory.get(current, 0.0) + 1.0
        scores = []
        for syn, direction in choices:
            node = syn.target if direction == "forward" else syn.source
            scores.append(self._memory.get(node, 0.0))
        idx = int(torch.tensor(scores).argmin().item())
        return choices[idx]


__all__ = ["MemoryDecayPlugin"]

PLUGIN_NAME = "memorydecay"

