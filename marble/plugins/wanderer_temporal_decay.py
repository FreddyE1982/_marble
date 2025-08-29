from __future__ import annotations

"""Temporal decay wanderer plugin.

Exploration probability decays exponentially with step count using a
learnable decay rate.
"""

import math
import random
from typing import List, Tuple

from ..wanderer import register_wanderer_type, expose_learnable_params


class TemporalDecayPlugin:
    """Decrease exploration over time."""

    @staticmethod
    @expose_learnable_params
    def _params(wanderer, *, decay_rate: float = 0.1):
        return (decay_rate,)

    def on_init(self, wanderer) -> None:
        wanderer._plugin_state["steps"] = 0

    def choose_next(self, wanderer, current, choices: List[Tuple["Synapse", str]]):
        if not choices:
            return None, "forward"
        (decay_t,) = self._params(wanderer)
        decay = float(decay_t.detach().to("cpu").item()) if hasattr(decay_t, "detach") else float(decay_t)
        step = wanderer._plugin_state.get("steps", 0)
        wanderer._plugin_state["steps"] = step + 1
        prob = math.exp(-decay * step)
        if random.random() < prob:
            return random.choice(choices)
        return max(choices, key=lambda cd: float(getattr(cd[0], "weight", 1.0)))


try:  # pragma: no cover
    register_wanderer_type("temporaldecay", TemporalDecayPlugin())
except Exception:
    pass

__all__ = ["TemporalDecayPlugin"]
