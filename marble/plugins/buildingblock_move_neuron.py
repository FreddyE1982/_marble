from __future__ import annotations

"""BuildingBlock: move a neuron to a new index."""

from typing import Sequence

from ..buildingblock import BuildingBlock
from ..wanderer import expose_learnable_params


class MoveNeuronPlugin(BuildingBlock):
    def _key(self, brain, index: Sequence[int]):
        return self._to_index(brain, index)

    @expose_learnable_params
    def apply(self, brain, old_index: Sequence[int], new_index: Sequence[int]):
        old_key = self._key(brain, old_index)
        new_key = self._key(brain, new_index)
        neuron = brain.neurons.get(old_key)
        if neuron is None or new_key in brain.neurons:
            return None
        brain.neurons.pop(old_key)
        setattr(neuron, "position", new_key)
        brain.neurons[new_key] = neuron
        return neuron


__all__ = ["MoveNeuronPlugin"]
