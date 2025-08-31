from __future__ import annotations

"""BuildingBlock: apply dropout to a neuron's tensor."""

import random
from typing import Sequence

from ..buildingblock import BuildingBlock
from ..wanderer import expose_learnable_params


class DropoutNeuronTensorPlugin(BuildingBlock):
    @expose_learnable_params
    def apply(self, brain, index: Sequence[int], p: float = 0.5):
        idx = self._to_index(brain, index)
        neuron = brain.get_neuron(idx)
        torch = self._torch
        if neuron is None:
            return None
        prob = self._to_float(p)
        if torch is not None and self._is_torch_tensor(neuron.tensor):
            mask = (torch.rand_like(neuron.tensor) > prob).to(neuron.tensor.dtype)
            neuron.tensor = neuron.tensor * mask
        else:
            tensor_list = neuron.tensor if isinstance(neuron.tensor, list) else list(neuron.tensor)
            mask = [1.0 if random.random() > prob else 0.0 for _ in tensor_list]
            neuron.tensor = [float(v) * m for v, m in zip(tensor_list, mask)]
        return neuron.tensor


__all__ = ["DropoutNeuronTensorPlugin"]
