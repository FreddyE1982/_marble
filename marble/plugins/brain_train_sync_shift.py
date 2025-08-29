"""SyncShiftTrainPlugin offsets start neuron using a learnable phase shift.

The ``phase_shift`` parameter, exposed via ``expose_learnable_params``,
determines how many indices to rotate when choosing the starting neuron for a
walk. This introduces a deterministic yet learnable curriculum over available
neurons.
"""

from __future__ import annotations

from typing import Any, Dict

from ..wanderer import expose_learnable_params
from ..reporter import report


@expose_learnable_params
def _phase_param(wanderer, phase_shift: float = 0.0):
    return phase_shift


class SyncShiftTrainPlugin:
    """Brain-train plugin selecting start neuron via learnable phase shift."""

    def choose_start(self, brain: "Brain", wanderer: "Wanderer", i: int):
        phase_t = _phase_param(wanderer)
        try:
            phase = int(float(phase_t.detach().to("cpu").item()))
        except Exception:
            phase = 0
        avail = list(brain.available_indices())
        if not avail:
            return None
        idx = avail[(phase + i) % len(avail)]
        report("training", "sync_shift_choose", {"walk": i, "index": idx}, "brain")
        return brain.neurons.get(idx)


__all__ = ["SyncShiftTrainPlugin"]

