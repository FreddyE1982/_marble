from __future__ import annotations

"""Delay synapse plugin.

Implements a first-order IIR low-pass filter (exponential moving average) with a
learnable blending factor controlling how much of the previous output is mixed
into the current transmission.
"""

from typing import Any, List

from ..graph import Synapse
from ..wanderer import expose_learnable_params
from ..reporter import report


class DelaySynapsePlugin:
    """Exponential moving average across transmissions."""

    def __init__(self) -> None:
        self._prev = {}

    @staticmethod
    @expose_learnable_params
    def _params(wanderer, *, delay_alpha: float = 0.5) -> Any:
        return (delay_alpha,)

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
        alpha = 0.5
        if wanderer is not None:
            (alpha,) = self._params(wanderer)
        alpha_f = float(alpha.detach().to("cpu").item()) if hasattr(alpha, "detach") else float(alpha)

        prev = self._prev.get(id(syn), 0.0)
        vals = self._to_list(value)
        out_vals: List[float] = []
        for v in vals:
            prev = alpha_f * prev + (1.0 - alpha_f) * v
            out_vals.append(prev)
        self._prev[id(syn)] = prev
        out = out_vals if len(out_vals) != 1 else out_vals[0]

        try:
            report(
                "synapse",
                "delay_step",
                {"alpha": alpha_f},
                "plugins",
            )
        except Exception:
            pass

        orig = syn.type_name
        syn.type_name = None
        try:
            return Synapse.transmit(syn, out, direction=direction)
        finally:
            syn.type_name = orig

__all__ = ["DelaySynapsePlugin"]

