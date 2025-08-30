from __future__ import annotations

"""BuildingBlock: increment a neuron's age."""

from typing import Sequence

from ..buildingblock import BuildingBlock
from ..wanderer import expose_learnable_params


class IncrementNeuronAgePlugin(BuildingBlock):
    @expose_learnable_params
    def apply(self, brain, index: Sequence[int], delta: int = 1) -> int | None:
        idx = self._to_index(brain, index)
        neuron = brain.get_neuron(idx)
        if neuron is None:
            return None
        neuron.step_age(int(self._to_float(delta)))
        return neuron.age


__all__ = ["IncrementNeuronAgePlugin"]
