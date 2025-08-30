from __future__ import annotations

"""BuildingBlock: scale a synapse's weight by a factor."""

from ..buildingblock import BuildingBlock
from ..wanderer import expose_learnable_params


class ScaleSynapseWeightPlugin(BuildingBlock):
    @expose_learnable_params
    def apply(self, brain, synapse, factor: float) -> float | None:
        if synapse not in getattr(brain, "synapses", []):
            return None
        synapse.weight = self._to_float(synapse.weight) * self._to_float(factor)
        return synapse.weight


__all__ = ["ScaleSynapseWeightPlugin"]
