from __future__ import annotations
from typing import Any
from ..wanderer import expose_learnable_params
from ..reporter import report

@expose_learnable_params
def _scale_param(wanderer, scale: float = 1.01):
    return scale,

class SynapseScalerPlugin:
    """Multiply traversed synapse weights by a learnable scale."""

    def on_step(self, wanderer: "Wanderer", current: "Neuron", syn: "Synapse", direction: str, step_index: int, out_value: Any) -> None:
        scale_t, = _scale_param(wanderer)
        try:
            scale = float(scale_t.detach().to("cpu").item())
        except Exception:
            scale = 1.0
        try:
            syn.weight = float(getattr(syn, "weight", 1.0)) * scale
            report("neuroplasticity", "scale", {"step": int(step_index), "scale": scale}, "plugins")
        except Exception:
            pass
        return None

__all__ = ["SynapseScalerPlugin"]
