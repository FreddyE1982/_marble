from __future__ import annotations

"""BuildingBlock: change synapse type."""

from ..buildingblock import BuildingBlock
from ..graph import _SYNAPSE_TYPES, Synapse
from ..wanderer import expose_learnable_params


class ChangeSynapseTypePlugin(BuildingBlock):
    @expose_learnable_params
    def apply(self, brain, synapse: Synapse, type_name: str):
        if synapse not in getattr(brain, "synapses", []):
            return None
        synapse.type_name = type_name
        plugin = _SYNAPSE_TYPES.get(type_name)
        if plugin is not None and hasattr(plugin, "on_init"):
            plugin.on_init(synapse)  # type: ignore[attr-defined]
        return synapse.type_name


__all__ = ["ChangeSynapseTypePlugin"]
