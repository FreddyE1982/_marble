from __future__ import annotations

"""BuildingBlock: change synapse bias."""

from ..buildingblock import BuildingBlock
from ..graph import Synapse


class ChangeSynapseBiasPlugin(BuildingBlock):
    def apply(self, brain, synapse: Synapse, bias: float):
        synapse.bias = float(bias)
        return synapse.bias


__all__ = ["ChangeSynapseBiasPlugin"]
