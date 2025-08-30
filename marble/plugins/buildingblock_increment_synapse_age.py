from __future__ import annotations

"""BuildingBlock: increment a synapse's age."""

from ..buildingblock import BuildingBlock
from ..wanderer import expose_learnable_params


class IncrementSynapseAgePlugin(BuildingBlock):
    @expose_learnable_params
    def apply(self, brain, synapse, delta: int = 1) -> int | None:
        if synapse not in getattr(brain, "synapses", []):
            return None
        synapse.step_age(int(self._to_float(delta)))
        return synapse.age


__all__ = ["IncrementSynapseAgePlugin"]
