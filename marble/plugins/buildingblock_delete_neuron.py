from __future__ import annotations

"""BuildingBlock: delete a neuron from the brain."""

from typing import Sequence

from ..buildingblock import BuildingBlock


class DeleteNeuronPlugin(BuildingBlock):
    def apply(self, brain, index: Sequence[int]) -> bool:
        neuron = brain.get_neuron(index)
        if neuron is None:
            raise ValueError("Neuron not found")
        brain.remove_neuron(neuron)
        return True


__all__ = ["DeleteNeuronPlugin"]
