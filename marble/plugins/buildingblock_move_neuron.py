from __future__ import annotations

"""BuildingBlock: move a neuron to a new index."""

from typing import Sequence

from ..buildingblock import BuildingBlock


class MoveNeuronPlugin(BuildingBlock):
    def _key(self, brain, index: Sequence[int]):
        if brain.mode == "grid":
            return tuple(int(i) for i in index)
        return tuple(float(i) for i in index)

    def apply(self, brain, old_index: Sequence[int], new_index: Sequence[int]):
        old_key = self._key(brain, old_index)
        new_key = self._key(brain, new_index)
        if new_key in brain.neurons:
            raise ValueError("Destination already occupied")
        neuron = brain.neurons.pop(old_key)
        setattr(neuron, "position", new_key)
        brain.neurons[new_key] = neuron
        return neuron


__all__ = ["MoveNeuronPlugin"]
