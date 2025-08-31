from __future__ import annotations

"""BuildingBlock: reset a neuron's tensor to zeros."""

from typing import Sequence

from ..buildingblock import BuildingBlock
from ..wanderer import expose_learnable_params


class ResetNeuronTensorPlugin(BuildingBlock):
    @expose_learnable_params
    def apply(self, brain, index: Sequence[int]):
        idx = self._to_index(brain, index)
        neuron = brain.get_neuron(idx)
        if neuron is None:
            return None
        torch = self._torch
        if torch is not None and self._is_torch_tensor(neuron.tensor):
            neuron.tensor = torch.zeros_like(neuron.tensor)
        else:
            tensor_list = neuron.tensor if isinstance(neuron.tensor, list) else list(neuron.tensor)
            neuron.tensor = [0.0 for _ in tensor_list]
        return neuron.tensor


__all__ = ["ResetNeuronTensorPlugin"]
