from __future__ import annotations

"""BuildingBlock: scale a neuron's bias by a factor."""

from typing import Sequence

from ..buildingblock import BuildingBlock


class ScaleNeuronBiasPlugin(BuildingBlock):
    def apply(self, brain, index: Sequence[int], factor: float) -> float:
        neuron = brain.get_neuron(index)
        if neuron is None:
            raise ValueError("Neuron not found")
        neuron.bias = float(neuron.bias) * float(factor)
        return neuron.bias


__all__ = ["ScaleNeuronBiasPlugin"]
