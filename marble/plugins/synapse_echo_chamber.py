from __future__ import annotations

"""Echo chamber synapse plugin.

Creates a decaying echo of past transmissions so current values are
interwoven with memory of earlier signals. Two learnable parameters control
the decay rate and the effective depth of the echo.
"""

from typing import Any, List

from ..graph import Synapse
from ..wanderer import expose_learnable_params
from ..reporter import report


class EchoChamberSynapsePlugin:
    """Applies a recursive echo to transmissions."""

    def __init__(self) -> None:
        self._echo = {}

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer,
        *,
        echo_decay: float = 0.5,
        echo_depth: float = 3.0,
    ) -> Any:
        return (echo_decay, echo_depth)

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
        decay, depth = 0.5, 3.0
        if wanderer is not None:
            decay, depth = self._params(wanderer)
        decay_f = float(decay.detach().to("cpu").item()) if hasattr(decay, "detach") else float(decay)
        depth_i = int(
            float(depth.detach().to("cpu").item()) if hasattr(depth, "detach") else float(depth)
        )

        prev = self._echo.get(id(syn), 0.0)
        vals = self._to_list(value)
        out_vals: List[float] = []
        for v in vals:
            h = prev
            for _ in range(max(1, depth_i)):
                h = v + h * decay_f
            prev = h
            out_vals.append(h)
        self._echo[id(syn)] = prev
        out = out_vals if len(out_vals) != 1 else out_vals[0]

        try:
            report(
                "synapse",
                "echo_step",
                {"decay": decay_f, "depth": depth_i},
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


__all__ = ["EchoChamberSynapsePlugin"]

