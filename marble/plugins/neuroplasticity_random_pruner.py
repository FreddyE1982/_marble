from __future__ import annotations
import random
from typing import Any, Dict
from ..wanderer import expose_learnable_params
from ..reporter import report

@expose_learnable_params
def _prune_param(wanderer, prune_threshold: int = 2):
    return prune_threshold,

class RandomPrunerPlugin:
    """Randomly prune an outgoing synapse if too many exist."""

    def on_walk_end(self, wanderer: "Wanderer", stats: Dict[str, Any]) -> None:
        thresh_t, = _prune_param(wanderer)
        try:
            thresh = int(float(thresh_t.detach().to("cpu").item()))
        except Exception:
            thresh = 2
        last = wanderer._visited[-1] if getattr(wanderer, "_visited", []) else None
        if last is None:
            return None
        outgoing = getattr(last, "outgoing", []) or []
        if len(outgoing) <= thresh:
            return None
        syn = random.choice(outgoing)
        try:
            wanderer.brain.remove_synapse(syn)
            report("neuroplasticity", "prune", {"from": getattr(last, "position", None)}, "plugins")
        except Exception:
            pass
        return None

__all__ = ["RandomPrunerPlugin"]
