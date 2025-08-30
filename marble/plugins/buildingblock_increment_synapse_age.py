from __future__ import annotations

"""BuildingBlock: increment a synapse's age."""

from ..buildingblock import BuildingBlock


class IncrementSynapseAgePlugin(BuildingBlock):
    def apply(self, brain, synapse, delta: int = 1) -> int:
        if synapse not in brain.synapses:
            raise ValueError("Synapse not in brain")
        synapse.step_age(delta)
        return synapse.age


__all__ = ["IncrementSynapseAgePlugin"]
