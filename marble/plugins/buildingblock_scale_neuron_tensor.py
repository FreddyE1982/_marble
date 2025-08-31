from __future__ import annotations

"""BuildingBlock: scale a neuron's tensor by a factor."""

from typing import Sequence

from ..buildingblock import BuildingBlock
from ..wanderer import expose_learnable_params


class ScaleNeuronTensorPlugin(BuildingBlock):
    @expose_learnable_params
    def apply(self, brain, index: Sequence[int], factor: float = 1.0):
        idx = self._to_index(brain, index)
        neuron = brain.get_neuron(idx)
        if neuron is None:
            return None
        torch = self._torch
        fac = self._to_float(factor)
        if torch is not None and self._is_torch_tensor(neuron.tensor):
            neuron.tensor = neuron.tensor * fac
        else:
            tensor_list = neuron.tensor if isinstance(neuron.tensor, list) else list(neuron.tensor)
            neuron.tensor = [float(v) * fac for v in tensor_list]
        return neuron.tensor


__all__ = ["ScaleNeuronTensorPlugin"]
