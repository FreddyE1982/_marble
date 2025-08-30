from __future__ import annotations

"""BuildingBlock: scale a neuron's weight by a factor."""

from typing import Sequence

from ..buildingblock import BuildingBlock


class ScaleNeuronWeightPlugin(BuildingBlock):
    def apply(self, brain, index: Sequence[int], factor: float) -> float:
        neuron = brain.get_neuron(index)
        if neuron is None:
            raise ValueError("Neuron not found")
        neuron.weight = float(neuron.weight) * float(factor)
        return neuron.weight


__all__ = ["ScaleNeuronWeightPlugin"]
