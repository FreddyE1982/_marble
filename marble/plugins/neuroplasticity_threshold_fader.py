"""ThresholdFaderPlugin attenuates synapse weights with a learnable rate."""

from __future__ import annotations

from typing import Any

from ..wanderer import expose_learnable_params
from ..reporter import report


@expose_learnable_params
def _fade_param(wanderer, fade_rate: float = 0.05):
    return fade_rate


class ThresholdFaderPlugin:
    """Fade synapse weights using an exponential decay."""

    def on_step(self, wanderer: "Wanderer", current: "Neuron", syn: "Synapse", direction: str, step_index: int, out_value: Any) -> None:
        fr_t = _fade_param(wanderer)
        try:
            fr = float(fr_t.detach().to("cpu").item())
        except Exception:
            fr = 0.0
        try:
            syn.weight = float(getattr(syn, "weight", 1.0)) * (1.0 - fr)
            report("neuroplasticity", "threshold_fade", {"step": int(step_index), "fade": fr}, "plugins")
        except Exception:
            pass
        return None


__all__ = ["ThresholdFaderPlugin"]

