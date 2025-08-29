from __future__ import annotations

"""Nonlocal tunnel synapse plugin.

Adds a persistent random offset to every transmission, modelling a
nonlocal shortcut that shifts signals through an imagined tunnel. The
strength of the tunnel effect is learnable.
"""

import random
from typing import Any, List

from ..graph import Synapse
from ..wanderer import expose_learnable_params
from ..reporter import report


class NonlocalTunnelSynapsePlugin:
    """Injects a fixed random offset into transmissions."""

    def __init__(self) -> None:
        self._offsets = {}

    @staticmethod
    @expose_learnable_params
    def _params(wanderer, *, tunnel_strength: float = 0.1) -> Any:
        return (tunnel_strength,)

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
        strength = 0.1
        if wanderer is not None:
            (strength,) = self._params(wanderer)
        strength_f = (
            float(strength.detach().to("cpu").item()) if hasattr(strength, "detach") else float(strength)
        )

        off = self._offsets.setdefault(id(syn), random.uniform(-1.0, 1.0))
        vals = self._to_list(value)
        out_vals = [v + off * strength_f for v in vals]
        out = out_vals if len(out_vals) != 1 else out_vals[0]

        try:
            report(
                "synapse",
                "nonlocal_shift",
                {"strength": strength_f},
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


__all__ = ["NonlocalTunnelSynapsePlugin"]

