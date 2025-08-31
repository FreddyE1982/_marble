from __future__ import annotations

"""BuildingBlock: clamp a neuron's tensor to a range."""

from typing import Sequence

from ..buildingblock import BuildingBlock
from ..wanderer import expose_learnable_params


class ClampNeuronTensorPlugin(BuildingBlock):
    @expose_learnable_params
    def apply(self, brain, index: Sequence[int], min_val: float = -1.0, max_val: float = 1.0):
        idx = self._to_index(brain, index)
        neuron = brain.get_neuron(idx)
        if neuron is None:
            return None
        mn = self._to_float(min_val)
        mx = self._to_float(max_val)
        torch = self._torch
        if torch is not None and self._is_torch_tensor(neuron.tensor):
            neuron.tensor = neuron.tensor.clamp(mn, mx)
        else:
            tensor_list = neuron.tensor if isinstance(neuron.tensor, list) else list(neuron.tensor)
            neuron.tensor = [float(min(max(v, mn), mx)) for v in tensor_list]
        return neuron.tensor


__all__ = ["ClampNeuronTensorPlugin"]
