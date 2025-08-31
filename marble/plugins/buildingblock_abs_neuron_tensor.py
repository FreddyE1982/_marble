from __future__ import annotations

"""BuildingBlock: take absolute value of a neuron's tensor."""

from typing import Sequence

from ..buildingblock import BuildingBlock
from ..wanderer import expose_learnable_params


class AbsNeuronTensorPlugin(BuildingBlock):
    @expose_learnable_params
    def apply(self, brain, index: Sequence[int]):
        idx = self._to_index(brain, index)
        neuron = brain.get_neuron(idx)
        torch = self._torch
        if neuron is None:
            return None
        if torch is not None and self._is_torch_tensor(neuron.tensor):
            neuron.tensor = torch.abs(neuron.tensor)
        else:
            tensor_list = neuron.tensor if isinstance(neuron.tensor, list) else list(neuron.tensor)
            neuron.tensor = [abs(float(v)) for v in tensor_list]
        return neuron.tensor


__all__ = ["AbsNeuronTensorPlugin"]
