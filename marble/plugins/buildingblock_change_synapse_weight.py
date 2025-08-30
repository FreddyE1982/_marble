from __future__ import annotations

"""BuildingBlock: change synapse weight."""

from ..buildingblock import BuildingBlock
from ..graph import Synapse


class ChangeSynapseWeightPlugin(BuildingBlock):
    def apply(self, brain, synapse: Synapse, weight: float):
        synapse.weight = float(weight)
        return synapse.weight


__all__ = ["ChangeSynapseWeightPlugin"]
