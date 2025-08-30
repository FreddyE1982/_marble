from __future__ import annotations

"""BuildingBlock: delete a neuron from the brain."""

from typing import Sequence

from ..buildingblock import BuildingBlock
from ..wanderer import expose_learnable_params


class DeleteNeuronPlugin(BuildingBlock):
    @expose_learnable_params
    def apply(self, brain, index: Sequence[int]) -> bool:
        idx = self._to_index(brain, index)
        neuron = brain.get_neuron(idx)
        if neuron is None:
            return False
        brain.remove_neuron(neuron)
        return True


__all__ = ["DeleteNeuronPlugin"]
