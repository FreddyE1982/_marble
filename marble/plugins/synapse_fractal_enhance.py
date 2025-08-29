from __future__ import annotations

"""Fractal enhance synapse plugin.

Iteratively perturbs transmissions using a sinusoidal function to create a
fractal-like cascade. Depth and scale of the perturbation are learnable,
allowing the synapse to sculpt intricate responses.
"""

import math
from typing import Any, List

from ..graph import Synapse
from ..wanderer import expose_learnable_params
from ..reporter import report


class FractalEnhanceSynapsePlugin:
    """Applies repeated sinusoidal boosts to signals."""

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer,
        *,
        fract_depth: float = 2.0,
        fract_scale: float = 0.5,
    ) -> Any:
        return (fract_depth, fract_scale)

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
        depth, scale = 2.0, 0.5
        if wanderer is not None:
            depth, scale = self._params(wanderer)
        depth_i = int(
            float(depth.detach().to("cpu").item()) if hasattr(depth, "detach") else float(depth)
        )
        scale_f = float(scale.detach().to("cpu").item()) if hasattr(scale, "detach") else float(scale)

        vals = self._to_list(value)
        out_vals: List[float] = []
        for v in vals:
            x = v
            for _ in range(max(1, depth_i)):
                x = x + math.sin(x * scale_f)
            out_vals.append(x)
        out = out_vals if len(out_vals) != 1 else out_vals[0]

        try:
            report(
                "synapse",
                "fractal_enhance",
                {"depth": depth_i, "scale": scale_f},
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


__all__ = ["FractalEnhanceSynapsePlugin"]

