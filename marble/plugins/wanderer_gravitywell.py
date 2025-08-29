from __future__ import annotations

"""Gravity-well biased wanderer plugin."""

import torch
from typing import List, Tuple

from ..wanderer import expose_learnable_params


class GravityWellPlugin:
    """Biases selection toward spatially central neurons."""

    @staticmethod
    @expose_learnable_params
    def _params(wanderer, *, gravity_strength: float = 1.0):
        return (gravity_strength,)

    def choose_next(self, wanderer, current, choices: List[Tuple["Synapse", str]]):
        if not choices:
            return None, "forward"
        (g_t,) = self._params(wanderer)
        g = float(g_t.detach().to("cpu").item()) if hasattr(g_t, "detach") else float(g_t)
        dists = []
        for syn, direction in choices:
            node = syn.target if direction == "forward" else syn.source
            pos = getattr(node, "position", (0.0,))
            tensor_pos = torch.tensor(list(pos), dtype=torch.float32)
            dists.append(torch.norm(tensor_pos))
        dists_t = torch.stack(dists)
        weights = torch.exp(-g * dists_t)
        probs = weights / weights.sum()
        idx = int(torch.multinomial(probs, 1).item())
        return choices[idx]


__all__ = ["GravityWellPlugin"]

PLUGIN_NAME = "gravitywell"

