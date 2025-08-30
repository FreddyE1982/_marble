"""EchoRepeaterTrainPlugin reuses a learnable index to start each walk.

The ``echo_strength`` parameter determines which neuron to revisit at the
beginning of every walk, creating a resonant training pattern that may reveal
subtle basin geometries.
"""

from __future__ import annotations

from typing import Any

from ..wanderer import expose_learnable_params
from ..reporter import report


@expose_learnable_params
def _echo_param(wanderer, echo_strength: float = 0.0):
    return echo_strength


class EchoRepeaterTrainPlugin:
    """Brain-train plugin reusing a learnable start index."""

    def choose_start(self, brain: "Brain", wanderer: "Wanderer", i: int):
        echo_t = _echo_param(wanderer)
        try:
            echo = int(float(echo_t.detach().to("cpu").item()))
        except Exception:
            echo = 0
        avail = list(brain.available_indices())
        if not avail:
            return None
        idx = avail[echo % len(avail)]
        report("training", "echo_repeater", {"walk": i, "index": idx}, "brain")
        return brain.neurons.get(idx)


__all__ = ["EchoRepeaterTrainPlugin"]

