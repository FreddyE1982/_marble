from __future__ import annotations

"""BuildingBlock: exponentiate a neuron's tensor."""

import math
from typing import Sequence

from ..buildingblock import BuildingBlock
from ..wanderer import expose_learnable_params


class ExpNeuronTensorPlugin(BuildingBlock):
    @expose_learnable_params
    def apply(self, brain, index: Sequence[int]):
        idx = self._to_index(brain, index)
        neuron = brain.get_neuron(idx)
        torch = self._torch
        if neuron is None:
            return None
        if torch is not None and self._is_torch_tensor(neuron.tensor):
            neuron.tensor = torch.exp(neuron.tensor)
        else:
            tensor_list = neuron.tensor if isinstance(neuron.tensor, list) else list(neuron.tensor)
            neuron.tensor = [math.exp(float(v)) for v in tensor_list]
        return neuron.tensor


__all__ = ["ExpNeuronTensorPlugin"]
