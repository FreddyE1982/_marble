from __future__ import annotations

"""BuildingBlock: change neuron weight."""

from typing import Sequence

from ..buildingblock import BuildingBlock


class ChangeNeuronWeightPlugin(BuildingBlock):
    def apply(self, brain, index: Sequence[int], weight: float):
        neuron = brain.get_neuron(index)
        if neuron is None:
            raise ValueError("Neuron not found")
        neuron.weight = float(weight)
        return neuron.weight


__all__ = ["ChangeNeuronWeightPlugin"]
