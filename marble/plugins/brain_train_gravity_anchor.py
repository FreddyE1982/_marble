"""GravityAnchorTrainPlugin shortens walks with a learnable gravity bias.

The ``gravity_bias`` parameter pulls walk lengths toward a base by inverting
their magnitude, simulating a gravity well that resists exploration and could
stabilise chaotic optimisation dynamics.
"""

from __future__ import annotations

from typing import Any, Dict

from ..wanderer import expose_learnable_params
from ..reporter import report


@expose_learnable_params
def _gravity_param(wanderer, gravity_bias: float = 0.0):
    return gravity_bias


class GravityAnchorTrainPlugin:
    """Brain-train plugin adjusting steps via a gravity-like bias."""

    def before_walk(self, brain: "Brain", wanderer: "Wanderer", i: int) -> Dict[str, Any]:
        g_t = _gravity_param(wanderer)
        try:
            g = float(g_t.detach().to("cpu").item())
        except Exception:
            g = 0.0
        scale = 1.0 / (1.0 + abs(g))
        steps = max(1, int(scale * (i + 1)))
        report("training", "gravity_anchor", {"walk": i, "steps": steps}, "brain")
        return {"max_steps": steps}


__all__ = ["GravityAnchorTrainPlugin"]

