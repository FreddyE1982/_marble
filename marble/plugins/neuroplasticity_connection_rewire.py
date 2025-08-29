from __future__ import annotations
import random
from typing import Any
from ..wanderer import expose_learnable_params
from ..reporter import report

@expose_learnable_params
def _rewire_params(wanderer, rewire_prob: float = 0.5):
    return rewire_prob,

class ConnectionRewirePlugin:
    """Occasionally connect current neuron to a random free index."""

    def on_step(self, wanderer: "Wanderer", current: "Neuron", syn: "Synapse", direction: str, step_index: int, out_value: Any) -> None:
        prob_t, = _rewire_params(wanderer)
        try:
            prob = float(prob_t.detach().to("cpu").item())
        except Exception:
            prob = 0.5
        if random.random() >= prob:
            return None
        brain = wanderer.brain
        try:
            avail = list(brain.available_indices())
        except Exception:
            avail = []
        if not avail:
            return None
        new_idx = random.choice(avail)
        try:
            brain.connect(getattr(current, "position"), new_idx, direction="uni")
            report("neuroplasticity", "rewire", {"from": getattr(current, "position", None), "to": new_idx}, "plugins")
        except Exception:
            pass
        return None

__all__ = ["ConnectionRewirePlugin"]
