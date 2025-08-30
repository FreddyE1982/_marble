from __future__ import annotations

"""BuildingBlock: swap the positions of two neurons."""

from typing import Sequence, Tuple

from ..buildingblock import BuildingBlock
from ..wanderer import expose_learnable_params


class SwapNeuronsPlugin(BuildingBlock):
    @expose_learnable_params
    def apply(self, brain, index_a: Sequence[int], index_b: Sequence[int]) -> Tuple[Tuple[int, ...], Tuple[int, ...]] | None:
        idx_a = self._to_index(brain, index_a)
        idx_b = self._to_index(brain, index_b)
        neuron_a = brain.get_neuron(idx_a)
        neuron_b = brain.get_neuron(idx_b)
        if neuron_a is None or neuron_b is None:
            return None
        pos_a = getattr(neuron_a, "position")
        pos_b = getattr(neuron_b, "position")
        brain.neurons[pos_a], brain.neurons[pos_b] = neuron_b, neuron_a
        setattr(neuron_a, "position", pos_b)
        setattr(neuron_b, "position", pos_a)
        return pos_b, pos_a


__all__ = ["SwapNeuronsPlugin"]
