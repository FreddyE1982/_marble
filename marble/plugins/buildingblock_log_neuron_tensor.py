from __future__ import annotations

"""BuildingBlock: apply logarithm to a neuron's tensor."""

import math
from typing import Sequence

from ..buildingblock import BuildingBlock
from ..wanderer import expose_learnable_params


class LogNeuronTensorPlugin(BuildingBlock):
    @expose_learnable_params
    def apply(self, brain, index: Sequence[int], epsilon: float = 1e-6):
        idx = self._to_index(brain, index)
        neuron = brain.get_neuron(idx)
        torch = self._torch
        if neuron is None:
            return None
        eps = self._to_float(epsilon)
        if torch is not None and self._is_torch_tensor(neuron.tensor):
            neuron.tensor = torch.log(torch.abs(neuron.tensor) + eps)
        else:
            tensor_list = neuron.tensor if isinstance(neuron.tensor, list) else list(neuron.tensor)
            neuron.tensor = [math.log(abs(float(v)) + eps) for v in tensor_list]
        return neuron.tensor


__all__ = ["LogNeuronTensorPlugin"]
