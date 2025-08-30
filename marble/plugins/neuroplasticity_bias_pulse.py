"""BiasPulsePlugin injects a learnable pulse into neuron bias values."""

from __future__ import annotations

from typing import Any

from ..wanderer import expose_learnable_params
from ..reporter import report


@expose_learnable_params
def _pulse_param(wanderer, pulse_intensity: float = 0.1):
    return pulse_intensity


class BiasPulsePlugin:
    """Add a learnable pulse to neuron bias on each traversal."""

    def on_step(self, wanderer: "Wanderer", current: "Neuron", syn: "Synapse", direction: str, step_index: int, out_value: Any) -> None:
        p_t = _pulse_param(wanderer)
        try:
            pulse = float(p_t.detach().to("cpu").item())
        except Exception:
            pulse = 0.0
        try:
            if hasattr(current, "bias"):
                base = getattr(current, "bias", 0.0)
                if hasattr(base, "detach"):
                    base = float(base.detach().to("cpu").item())
                else:
                    base = float(base)
                current.bias = base + pulse
            report("neuroplasticity", "bias_pulse", {"step": int(step_index), "pulse": pulse}, "plugins")
        except Exception:
            pass
        return None


__all__ = ["BiasPulsePlugin"]

