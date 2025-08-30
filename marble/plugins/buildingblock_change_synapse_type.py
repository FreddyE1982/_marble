from __future__ import annotations

"""BuildingBlock: change synapse type."""

from ..buildingblock import BuildingBlock
from ..graph import _SYNAPSE_TYPES, Synapse


class ChangeSynapseTypePlugin(BuildingBlock):
    def apply(self, brain, synapse: Synapse, type_name: str):
        synapse.type_name = type_name
        plugin = _SYNAPSE_TYPES.get(type_name)
        if plugin is not None and hasattr(plugin, "on_init"):
            plugin.on_init(synapse)  # type: ignore[attr-defined]
        return synapse.type_name


__all__ = ["ChangeSynapseTypePlugin"]
