from __future__ import annotations

"""Causal loop synapse plugin.

Feeds part of the output back into the input as if time folded, allowing
self-referential signal loops that might drive emergent causality."""

from typing import Any, List

from ..graph import Synapse
from ..wanderer import expose_learnable_params
from ..reporter import report


class CausalLoopSynapsePlugin:
    """Inject a causal loop into transmission."""

    def __init__(self) -> None:
        self._last = {}

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer,
        *,
        loop_strength: float = 0.3,
    ):
        return (loop_strength,)

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
        strength = 0.3
        if wanderer is not None:
            (strength,) = self._params(wanderer)
        strength_f = float(strength.detach().to("cpu").item()) if hasattr(strength, "detach") else float(strength)

        prev = self._last.get(id(syn), 0.0)
        vals = self._to_list(value)
        out_vals = []
        for v in vals:
            out = v + prev * strength_f
            prev = out
            out_vals.append(out)
        self._last[id(syn)] = prev
        out = out_vals if len(out_vals) != 1 else out_vals[0]

        try:
            report("synapse", "causal_loop", {"strength": strength_f}, "plugins")
        except Exception:
            pass

        orig = syn.type_name
        syn.type_name = None
        try:
            return Synapse.transmit(syn, out, direction=direction)
        finally:
            syn.type_name = orig


__all__ = ["CausalLoopSynapsePlugin"]
