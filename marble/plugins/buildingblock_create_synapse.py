from __future__ import annotations

"""BuildingBlock: create a synapse."""

from typing import Sequence

from ..buildingblock import BuildingBlock


class CreateSynapsePlugin(BuildingBlock):
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
        return brain.connect(src_index, dst_index, direction=direction, weight=weight, bias=bias, type_name=type_name)


__all__ = ["CreateSynapsePlugin"]
