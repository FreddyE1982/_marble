from __future__ import annotations

"""BuildingBlock: shuffle the elements of a neuron's tensor."""

import random
from typing import Sequence

from ..buildingblock import BuildingBlock
from ..wanderer import expose_learnable_params


class ShuffleNeuronTensorPlugin(BuildingBlock):
    @expose_learnable_params
    def apply(self, brain, index: Sequence[int]):
        idx = self._to_index(brain, index)
        neuron = brain.get_neuron(idx)
        if neuron is None:
            return None
        torch = self._torch
        if torch is not None and self._is_torch_tensor(neuron.tensor):
            perm = torch.randperm(neuron.tensor.numel(), device=self._device)
            neuron.tensor = neuron.tensor[perm]
        else:
            tensor_list = neuron.tensor if isinstance(neuron.tensor, list) else list(neuron.tensor)
            random.shuffle(tensor_list)
            neuron.tensor = [float(v) for v in tensor_list]
        return neuron.tensor


__all__ = ["ShuffleNeuronTensorPlugin"]
