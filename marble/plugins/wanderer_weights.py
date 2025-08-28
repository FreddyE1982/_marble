from __future__ import annotations

from typing import List, Tuple

from ..wanderer import register_wanderer_type


class WanderAlongSynapseWeightsPlugin:
    """choose_next prefers the synapse with the highest weight among choices."""

    def choose_next(self, wanderer: "Wanderer", current: "Neuron", choices: List[Tuple["Synapse", str]]):
        if not choices:
            return None, "forward"
        best = choices[0]
        best_w = float(getattr(best[0], "weight", 1.0))
        for s, d in choices[1:]:
            w = float(getattr(s, "weight", 1.0))
            if w > best_w:
                best = (s, d)
                best_w = w
        return best


try:
    register_wanderer_type("wanderalongsynapseweights", WanderAlongSynapseWeightsPlugin())
except Exception:
    pass

__all__ = ["WanderAlongSynapseWeightsPlugin"]
