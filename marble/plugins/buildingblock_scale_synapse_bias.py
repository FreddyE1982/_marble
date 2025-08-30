from __future__ import annotations

"""BuildingBlock: scale a synapse's bias by a factor."""

from ..buildingblock import BuildingBlock


class ScaleSynapseBiasPlugin(BuildingBlock):
    def apply(self, brain, synapse, factor: float) -> float:
        if synapse not in brain.synapses:
            raise ValueError("Synapse not in brain")
        synapse.bias = float(synapse.bias) * float(factor)
        return synapse.bias


__all__ = ["ScaleSynapseBiasPlugin"]
