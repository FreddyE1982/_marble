from __future__ import annotations

"""BuildingBlock: add a constant shift to a neuron's tensor."""

from typing import Sequence

from ..buildingblock import BuildingBlock
from ..wanderer import expose_learnable_params


class ShiftNeuronTensorPlugin(BuildingBlock):
    @expose_learnable_params
    def apply(self, brain, index: Sequence[int], delta: float = 0.0):
        idx = self._to_index(brain, index)
        neuron = brain.get_neuron(idx)
        if neuron is None:
            return None
        torch = self._torch
        if torch is not None and self._is_torch_tensor(neuron.tensor):
            neuron.tensor = neuron.tensor + self._to_float(delta)
        else:
            tensor_list = neuron.tensor if isinstance(neuron.tensor, list) else list(neuron.tensor)
            neuron.tensor = [float(v) + self._to_float(delta) for v in tensor_list]
        return neuron.tensor


__all__ = ["ShiftNeuronTensorPlugin"]
