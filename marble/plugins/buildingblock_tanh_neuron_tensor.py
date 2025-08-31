from __future__ import annotations

"""BuildingBlock: apply tanh to a neuron's tensor."""

import math
from typing import Sequence

from ..buildingblock import BuildingBlock
from ..wanderer import expose_learnable_params


class TanhNeuronTensorPlugin(BuildingBlock):
    @expose_learnable_params
    def apply(self, brain, index: Sequence[int]):
        idx = self._to_index(brain, index)
        neuron = brain.get_neuron(idx)
        torch = self._torch
        if neuron is None:
            return None
        if torch is not None and self._is_torch_tensor(neuron.tensor):
            neuron.tensor = torch.tanh(neuron.tensor)
        else:
            tensor_list = neuron.tensor if isinstance(neuron.tensor, list) else list(neuron.tensor)
            neuron.tensor = [math.tanh(float(v)) for v in tensor_list]
        return neuron.tensor


__all__ = ["TanhNeuronTensorPlugin"]
