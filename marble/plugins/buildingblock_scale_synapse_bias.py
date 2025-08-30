from __future__ import annotations

"""BuildingBlock: scale a synapse's bias by a factor."""

from ..buildingblock import BuildingBlock
from ..wanderer import expose_learnable_params


class ScaleSynapseBiasPlugin(BuildingBlock):
    @expose_learnable_params
    def apply(self, brain, synapse, factor: float) -> float | None:
        if synapse not in getattr(brain, "synapses", []):
            return None
        synapse.bias = self._to_float(synapse.bias) * self._to_float(factor)
        return synapse.bias


__all__ = ["ScaleSynapseBiasPlugin"]
