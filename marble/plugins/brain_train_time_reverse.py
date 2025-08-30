"""TimeReverseTrainPlugin starts walks from the tail with a learnable bias.

Using ``reverse_bias`` the plugin selects neurons counting backward from the
end of available indices, effectively reversing temporal order and promoting
retrograde exploration.
"""

from __future__ import annotations

from typing import Any

from ..wanderer import expose_learnable_params
from ..reporter import report


@expose_learnable_params
def _reverse_param(wanderer, reverse_bias: float = 0.0):
    return reverse_bias


class TimeReverseTrainPlugin:
    """Brain-train plugin that selects start neurons from the end."""

    def choose_start(self, brain: "Brain", wanderer: "Wanderer", i: int):
        rev_t = _reverse_param(wanderer)
        try:
            rb = int(float(rev_t.detach().to("cpu").item()))
        except Exception:
            rb = 0
        avail = list(brain.available_indices())
        if not avail:
            return None
        idx = avail[-1 - (rb % len(avail))]
        report("training", "time_reverse", {"walk": i, "index": idx}, "brain")
        return brain.neurons.get(idx)


__all__ = ["TimeReverseTrainPlugin"]

