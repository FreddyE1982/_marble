from __future__ import annotations

"""Superposition surge synapse plugin.

Combines current and previous transmissions in a superposed surge intended to
create rich emergent signal interactions."""

from typing import Any, List

from ..graph import Synapse
from ..wanderer import expose_learnable_params
from ..reporter import report


class SuperpositionSurgeSynapsePlugin:
    """Fuse transmissions through superposition and surge memory."""

    def __init__(self) -> None:
        self._memory = {}

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer,
        *,
        surge_alpha: float = 0.5,
        surge_beta: float = 0.1,
    ):
        return (surge_alpha, surge_beta)

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
        alpha, beta = 0.5, 0.1
        if wanderer is not None:
            alpha, beta = self._params(wanderer)
        alpha_f = float(alpha.detach().to("cpu").item()) if hasattr(alpha, "detach") else float(alpha)
        beta_f = float(beta.detach().to("cpu").item()) if hasattr(beta, "detach") else float(beta)

        prev = self._memory.get(id(syn), 0.0)
        vals = self._to_list(value)
        out_vals = []
        for v in vals:
            surge = alpha_f * v + beta_f * prev
            prev = surge
            out_vals.append(surge)
        self._memory[id(syn)] = prev
        out = out_vals if len(out_vals) != 1 else out_vals[0]

        try:
            report("synapse", "superposition_surge", {"alpha": alpha_f, "beta": beta_f}, "plugins")
        except Exception:
            pass

        orig = syn.type_name
        syn.type_name = None
        try:
            return Synapse.transmit(syn, out, direction=direction)
        finally:
            syn.type_name = orig


__all__ = ["SuperpositionSurgeSynapsePlugin"]
