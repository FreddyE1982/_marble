from __future__ import annotations

"""BuildingBlock: normalize a neuron's tensor to unit norm."""

from typing import Sequence

from ..buildingblock import BuildingBlock
from ..wanderer import expose_learnable_params


class NormalizeNeuronTensorPlugin(BuildingBlock):
    @expose_learnable_params
    def apply(self, brain, index: Sequence[int]):
        idx = self._to_index(brain, index)
        neuron = brain.get_neuron(idx)
        if neuron is None:
            return None
        torch = self._torch
        if torch is not None and self._is_torch_tensor(neuron.tensor):
            norm = torch.linalg.vector_norm(neuron.tensor)
            if float(norm.detach().to("cpu").item()) > 0.0:
                neuron.tensor = neuron.tensor / norm
        else:
            tensor_list = neuron.tensor if isinstance(neuron.tensor, list) else list(neuron.tensor)
            norm = sum(float(v) ** 2 for v in tensor_list) ** 0.5
            if norm > 0.0:
                neuron.tensor = [float(v) / norm for v in tensor_list]
        return neuron.tensor


__all__ = ["NormalizeNeuronTensorPlugin"]
