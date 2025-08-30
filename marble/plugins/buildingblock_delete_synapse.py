from __future__ import annotations

"""BuildingBlock: delete a synapse from the brain."""

from ..buildingblock import BuildingBlock
from ..wanderer import expose_learnable_params


class DeleteSynapsePlugin(BuildingBlock):
    @expose_learnable_params
    def apply(self, brain, synapse) -> bool:
        if synapse not in getattr(brain, "synapses", []):
            return False
        brain.remove_synapse(synapse)
        return True


__all__ = ["DeleteSynapsePlugin"]
