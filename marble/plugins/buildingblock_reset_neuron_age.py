from __future__ import annotations

"""BuildingBlock: reset a neuron's age to zero."""

from typing import Sequence

from ..buildingblock import BuildingBlock
from ..wanderer import expose_learnable_params


class ResetNeuronAgePlugin(BuildingBlock):
    @expose_learnable_params
    def apply(self, brain, index: Sequence[int]):
        idx = self._to_index(brain, index)
        neuron = brain.get_neuron(idx)
        if neuron is None:
            return None
        neuron.age = 0
        return neuron.age


__all__ = ["ResetNeuronAgePlugin"]
