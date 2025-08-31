from __future__ import annotations

"""BuildingBlock: add Gaussian noise to a neuron's tensor."""

from typing import Sequence

from ..buildingblock import BuildingBlock
from ..wanderer import expose_learnable_params


class NoiseNeuronTensorPlugin(BuildingBlock):
    @expose_learnable_params
    def apply(self, brain, index: Sequence[int], sigma: float = 0.1):
        idx = self._to_index(brain, index)
        neuron = brain.get_neuron(idx)
        torch = self._torch
        if neuron is None or torch is None:
            return None
        sig = self._to_float(sigma)
        if self._is_torch_tensor(neuron.tensor):
            noise = torch.randn_like(neuron.tensor) * sig
            neuron.tensor = neuron.tensor + noise
        else:
            tensor_list = neuron.tensor if isinstance(neuron.tensor, list) else list(neuron.tensor)
            noise = torch.randn(len(tensor_list), device=self._device) * sig
            neuron.tensor = [float(v) + float(n) for v, n in zip(tensor_list, noise.tolist())]
        return neuron.tensor


__all__ = ["NoiseNeuronTensorPlugin"]
