from __future__ import annotations

"""BuildingBlock: create a synapse."""

from typing import Sequence

from ..buildingblock import BuildingBlock
from ..wanderer import expose_learnable_params


class CreateSynapsePlugin(BuildingBlock):
    @expose_learnable_params
    def apply(
        self,
        brain,
        src_index: Sequence[int],
        dst_index: Sequence[int],
        *,
        direction: str = "uni",
        weight: float = 1.0,
        bias: float = 0.0,
        type_name: str | None = None,
    ):
        src = self._to_index(brain, src_index)
        dst = self._to_index(brain, dst_index)
        try:
            return brain.connect(
                src,
                dst,
                direction=direction,
                weight=self._to_float(weight),
                bias=self._to_float(bias),
                type_name=type_name,
            )
        except Exception:
            return None


__all__ = ["CreateSynapsePlugin"]
