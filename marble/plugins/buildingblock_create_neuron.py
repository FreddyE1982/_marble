from __future__ import annotations

"""BuildingBlock: create a neuron."""

from typing import Any, Sequence

from ..buildingblock import BuildingBlock


class CreateNeuronPlugin(BuildingBlock):
    def apply(
        self,
        brain,
        index: Sequence[int],
        tensor: Any,
        *,
        weight: float = 1.0,
        bias: float = 0.0,
        type_name: str | None = None,
    ):
        return brain.add_neuron(index, tensor=tensor, weight=weight, bias=bias, type_name=type_name)


__all__ = ["CreateNeuronPlugin"]
