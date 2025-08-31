from __future__ import annotations

"""BuildingBlock: apply softmax to a neuron's tensor."""

import math
from typing import Sequence

from ..buildingblock import BuildingBlock
from ..wanderer import expose_learnable_params


class SoftmaxNeuronTensorPlugin(BuildingBlock):
    @expose_learnable_params
    def apply(self, brain, index: Sequence[int], dim: int = 0):
        idx = self._to_index(brain, index)
        neuron = brain.get_neuron(idx)
        torch = self._torch
        if neuron is None:
            return None
        d = int(dim)
        if torch is not None and self._is_torch_tensor(neuron.tensor):
            neuron.tensor = torch.softmax(neuron.tensor, dim=d)
        else:
            tensor_list = neuron.tensor if isinstance(neuron.tensor, list) else list(neuron.tensor)
            if torch is not None:
                t = torch.tensor(tensor_list, device=self._device, dtype=torch.float32)
                neuron.tensor = torch.softmax(t, dim=0).tolist()
            else:
                exps = [math.exp(float(v)) for v in tensor_list]
                s = sum(exps)
                neuron.tensor = [e / s for e in exps]
        return neuron.tensor


__all__ = ["SoftmaxNeuronTensorPlugin"]
