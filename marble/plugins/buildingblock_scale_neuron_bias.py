from __future__ import annotations

"""BuildingBlock: scale a neuron's bias by a factor."""

from typing import Sequence

from ..buildingblock import BuildingBlock
from ..wanderer import expose_learnable_params


class ScaleNeuronBiasPlugin(BuildingBlock):
    @expose_learnable_params
    def apply(self, brain, index: Sequence[int], factor: float) -> float | None:
        idx = self._to_index(brain, index)
        neuron = brain.get_neuron(idx)
        if neuron is None:
            return None
        neuron.bias = self._to_float(neuron.bias) * self._to_float(factor)
        return neuron.bias


__all__ = ["ScaleNeuronBiasPlugin"]
