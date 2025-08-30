from __future__ import annotations

"""BuildingBlock: change neuron bias."""

from typing import Sequence

from ..buildingblock import BuildingBlock


class ChangeNeuronBiasPlugin(BuildingBlock):
    def apply(self, brain, index: Sequence[int], bias: float):
        neuron = brain.get_neuron(index)
        if neuron is None:
            raise ValueError("Neuron not found")
        neuron.bias = float(bias)
        return neuron.bias


__all__ = ["ChangeNeuronBiasPlugin"]
