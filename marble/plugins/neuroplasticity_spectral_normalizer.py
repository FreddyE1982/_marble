from __future__ import annotations
from typing import Any
from ..wanderer import expose_learnable_params
from ..reporter import report

@expose_learnable_params
def _norm_params(wanderer, norm_rate: float = 0.9):
    return norm_rate,

class SpectralNormalizerPlugin:
    """Pull synapse weights toward unit magnitude."""

    def on_step(self, wanderer: "Wanderer", current: "Neuron", syn: "Synapse", direction: str, step_index: int, out_value: Any) -> None:
        rate_t, = _norm_params(wanderer)
        try:
            rate = float(rate_t.detach().to("cpu").item())
        except Exception:
            rate = 0.9
        try:
            w = float(getattr(syn, "weight", 1.0))
            sign = 1.0 if w >= 0 else -1.0
            syn.weight = w * rate + sign * (1.0 - rate)
            report("neuroplasticity", "spectral", {"step": int(step_index), "rate": rate}, "plugins")
        except Exception:
            pass
        return None

__all__ = ["SpectralNormalizerPlugin"]
