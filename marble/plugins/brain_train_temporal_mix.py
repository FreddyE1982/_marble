"""TemporalMixTrainPlugin blends walk indices into step counts.

Using a learnable ``mix_ratio`` parameter, the plugin scales the current walk
index and mixes it with a base of one step to compute ``max_steps``. This allows
the training process itself to discover an optimal pacing across walks.
"""

from __future__ import annotations

from typing import Any, Dict

from ..wanderer import expose_learnable_params
from ..reporter import report


@expose_learnable_params
def _mix_param(wanderer, mix_ratio: float = 0.5):
    return mix_ratio


class TemporalMixTrainPlugin:
    """Brain-train plugin computing steps from a learnable walk mix."""

    def before_walk(self, brain: "Brain", wanderer: "Wanderer", i: int) -> Dict[str, Any]:
        mix_t = _mix_param(wanderer)
        try:
            mix = float(mix_t.detach().to("cpu").item())
        except Exception:
            mix = 0.5
        steps = 1 + int(abs(mix) * (i + 1))
        report("training", "temporal_mix_steps", {"walk": i, "steps": steps}, "brain")
        return {"max_steps": steps}


__all__ = ["TemporalMixTrainPlugin"]

