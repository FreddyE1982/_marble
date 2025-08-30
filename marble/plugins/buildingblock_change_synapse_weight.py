from __future__ import annotations

"""BuildingBlock: change synapse weight."""

from ..buildingblock import BuildingBlock
from ..graph import Synapse
from ..wanderer import expose_learnable_params


class ChangeSynapseWeightPlugin(BuildingBlock):
    @expose_learnable_params
    def apply(self, brain, synapse: Synapse, weight: float):
        if synapse not in getattr(brain, "synapses", []):
            return None
        synapse.weight = self._to_float(weight)
        return synapse.weight


__all__ = ["ChangeSynapseWeightPlugin"]
