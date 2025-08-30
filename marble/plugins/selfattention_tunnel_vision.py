"""TunnelVisionRoutine narrows temperature based on a focus parameter."""

from __future__ import annotations

from typing import Any, Dict

from ..wanderer import expose_learnable_params
from ..reporter import report


@expose_learnable_params
def _focus_param(wanderer, tunnel_focus: float = 1.0):
    return tunnel_focus


class TunnelVisionRoutine:
    """Reduce temperature using a learnable focus strength."""

    def after_step(self, selfattention: "SelfAttention", reporter_ro: Any, wanderer: "Wanderer", step_index: int, ctx: Dict[str, Any]):
        tf_t = _focus_param(wanderer)
        try:
            tf = float(tf_t.detach().to("cpu").item())
        except Exception:
            tf = 1.0
        try:
            base = float(selfattention.get_param("temperature", 1.0))
            selfattention.set_param("temperature", base / (1.0 + abs(tf)))
        except Exception:
            pass
        report("selfattention", "tunnel_vision", {"step": step_index, "focus": tf}, "events")
        return None


__all__ = ["TunnelVisionRoutine"]

