from __future__ import annotations

"""BuildingBlock: create a neuron."""

from typing import Any, Optional, Sequence

from ..buildingblock import BuildingBlock
from ..wanderer import expose_learnable_params


class CreateNeuronPlugin(BuildingBlock):
    @expose_learnable_params
    def apply(
        self,
        brain,
        index: Sequence[int],
        tensor: Any,
        *,
        weight: float = 1.0,
        bias: float = 0.0,
        type_name: str | None = None,
        connect_to_index: Optional[Sequence[int]] = None,
        direction: str = "bi",
    ):
        idx = self._to_index(brain, index)
        conn = self._to_index(brain, connect_to_index) if connect_to_index is not None else None
        if conn is None and brain.neurons:
            conn = next(iter(brain.neurons))
        try:
            return brain.add_neuron(
                idx,
                tensor=tensor,
                connect_to=conn,
                direction=direction,
                weight=self._to_float(weight),
                bias=self._to_float(bias),
                type_name=type_name,
            )
        except Exception:
            return None


__all__ = ["CreateNeuronPlugin"]
