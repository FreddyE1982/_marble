from __future__ import annotations

"""BuildingBlock: change synapse bias."""

from ..buildingblock import BuildingBlock
from ..graph import Synapse
from ..wanderer import expose_learnable_params


class ChangeSynapseBiasPlugin(BuildingBlock):
    @expose_learnable_params
    def apply(self, brain, synapse: Synapse, bias: float):
        if synapse not in getattr(brain, "synapses", []):
            return None
        synapse.bias = self._to_float(bias)
        return synapse.bias


__all__ = ["ChangeSynapseBiasPlugin"]
