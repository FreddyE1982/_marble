from __future__ import annotations

"""BuildingBlock: randomize a neuron's weight within a range."""

from typing import Sequence

from ..buildingblock import BuildingBlock
from ..wanderer import expose_learnable_params


class RandomizeNeuronWeightPlugin(BuildingBlock):
    @expose_learnable_params
    def apply(self, brain, index: Sequence[int], min_val: float = -1.0, max_val: float = 1.0):
        idx = self._to_index(brain, index)
        neuron = brain.get_neuron(idx)
        torch = self._torch
        if neuron is None or torch is None:
            return None
        mn = self._to_float(min_val)
        mx = self._to_float(max_val)
        neuron.weight = float(torch.empty(1, device=self._device).uniform_(mn, mx).item())
        return neuron.weight


__all__ = ["RandomizeNeuronWeightPlugin"]
