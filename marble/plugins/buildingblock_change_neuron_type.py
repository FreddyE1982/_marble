from __future__ import annotations

"""BuildingBlock: change neuron type."""

from typing import Sequence

from ..buildingblock import BuildingBlock
from ..graph import _NEURON_TYPES
from ..wanderer import expose_learnable_params


class ChangeNeuronTypePlugin(BuildingBlock):
    @expose_learnable_params
    def apply(self, brain, index: Sequence[int], type_name: str):
        idx = self._to_index(brain, index)
        neuron = brain.get_neuron(idx)
        if neuron is None:
            return None
        neuron.type_name = type_name
        plugin = _NEURON_TYPES.get(type_name)
        if plugin is not None and hasattr(plugin, "on_init"):
            plugin.on_init(neuron)  # type: ignore[attr-defined]
        return neuron.type_name


__all__ = ["ChangeNeuronTypePlugin"]
