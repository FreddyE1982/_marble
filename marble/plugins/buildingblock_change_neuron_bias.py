from __future__ import annotations

"""BuildingBlock: change neuron bias."""

from typing import Sequence

from ..buildingblock import BuildingBlock
from ..wanderer import expose_learnable_params


class ChangeNeuronBiasPlugin(BuildingBlock):
    @expose_learnable_params
    def apply(self, brain, index: Sequence[int], bias: float):
        idx = self._to_index(brain, index)
        neuron = brain.get_neuron(idx)
        if neuron is None:
            return None
        neuron.bias = self._to_float(bias)
        return neuron.bias


__all__ = ["ChangeNeuronBiasPlugin"]
