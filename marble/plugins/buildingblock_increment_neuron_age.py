from __future__ import annotations

"""BuildingBlock: increment a neuron's age."""

from typing import Sequence

from ..buildingblock import BuildingBlock


class IncrementNeuronAgePlugin(BuildingBlock):
    def apply(self, brain, index: Sequence[int], delta: int = 1) -> int:
        neuron = brain.get_neuron(index)
        if neuron is None:
            raise ValueError("Neuron not found")
        neuron.step_age(delta)
        return neuron.age


__all__ = ["IncrementNeuronAgePlugin"]
