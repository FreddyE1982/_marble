from __future__ import annotations

"""BuildingBlock: scale a synapse's weight by a factor."""

from ..buildingblock import BuildingBlock


class ScaleSynapseWeightPlugin(BuildingBlock):
    def apply(self, brain, synapse, factor: float) -> float:
        if synapse not in brain.synapses:
            raise ValueError("Synapse not in brain")
        synapse.weight = float(synapse.weight) * float(factor)
        return synapse.weight


__all__ = ["ScaleSynapseWeightPlugin"]
