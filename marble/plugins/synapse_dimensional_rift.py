from __future__ import annotations

"""Dimensional rift synapse plugin.

Creates a temporary rift that alters transmissions depending on a learnable
rift depth, encouraging emergent cross-dimensional behaviour."""

from typing import Any, List
import math

from ..graph import Synapse
from ..wanderer import expose_learnable_params
from ..reporter import report


class DimensionalRiftSynapsePlugin:
    """Warp signals through an abstract dimensional rift."""

    def __init__(self) -> None:
        self._phase = {}

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer,
        *,
        rift_depth: float = 1.0,
    ):
        return (rift_depth,)

    def _to_list(self, value: Any) -> List[float]:
        if hasattr(value, "detach") and hasattr(value, "tolist"):
            return [float(v) for v in value.detach().to("cpu").view(-1).tolist()]
        if isinstance(value, (list, tuple)):
            return [float(v) for v in value]
        return [float(value)]

    def transmit(self, syn: "Synapse", value: Any, *, direction: str = "forward") -> Any:
        wanderer = getattr(getattr(syn.source, "_plugin_state", {}), "get", lambda *_: None)("wanderer")
        if wanderer is None:
            wanderer = getattr(getattr(syn.target, "_plugin_state", {}), "get", lambda *_: None)("wanderer")
        depth = 1.0
        if wanderer is not None:
            (depth,) = self._params(wanderer)
        depth_f = float(depth.detach().to("cpu").item()) if hasattr(depth, "detach") else float(depth)

        phase = self._phase.get(id(syn), 0.0) + depth_f
        self._phase[id(syn)] = phase
        vals = self._to_list(value)
        out_vals = [v * math.cos(phase) - v * math.sin(phase) for v in vals]
        out = out_vals if len(out_vals) != 1 else out_vals[0]

        try:
            report("synapse", "dimensional_rift", {"depth": depth_f}, "plugins")
        except Exception:
            pass

        orig = syn.type_name
        syn.type_name = None
        try:
            return Synapse.transmit(syn, out, direction=direction)
        finally:
            syn.type_name = orig


__all__ = ["DimensionalRiftSynapsePlugin"]
