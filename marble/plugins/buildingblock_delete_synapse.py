from __future__ import annotations

"""BuildingBlock: delete a synapse from the brain."""

from ..buildingblock import BuildingBlock


class DeleteSynapsePlugin(BuildingBlock):
    def apply(self, brain, synapse) -> bool:
        if synapse not in brain.synapses:
            raise ValueError("Synapse not in brain")
        brain.remove_synapse(synapse)
        return True


__all__ = ["DeleteSynapsePlugin"]
