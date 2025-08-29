from __future__ import annotations

"""Spiral time neuron plugin.

Applies a logarithmic spiral transform to the input before a sine wave,
introducing a temporal-like distortion that depends on input magnitude.
"""

from typing import Any, Tuple
import math

from ..wanderer import expose_learnable_params
from ..reporter import report


class SpiralTimeNeuronPlugin:
    """Logarithmic spiral sine activation."""

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer: "Wanderer",
        *,
        spiral_freq: float = 1.0,
        spiral_phase: float = 0.0,
        spiral_bias: float = 0.0,
    ) -> Tuple[Any, Any, Any]:
        return (spiral_freq, spiral_phase, spiral_bias)

    def forward(self, neuron: "Neuron", input_value=None):
        x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)

        freq = 1.0
        phase = 0.0
        bias = 0.0
        wanderer = neuron._plugin_state.get("wanderer")
        if wanderer is not None:
            try:
                freq, phase, bias = self._params(wanderer)
            except Exception:
                pass

        torch = getattr(neuron, "_torch", None)
        if torch is not None and neuron._is_torch_tensor(x):
            y = torch.sin(freq * torch.log1p(torch.abs(x)) + phase) + bias
            try:
                report(
                    "neuron",
                    "spiral_time_forward",
                    {
                        "freq": float(freq.detach().to("cpu").item())
                        if hasattr(freq, "detach")
                        else float(freq),
                        "phase": float(phase.detach().to("cpu").item())
                        if hasattr(phase, "detach")
                        else float(phase),
                        "bias": float(bias.detach().to("cpu").item())
                        if hasattr(bias, "detach")
                        else float(bias),
                    },
                    "plugins",
                )
            except Exception:
                pass
            return y

        x_list = x if isinstance(x, list) else [float(x)]
        freq_f = float(freq.detach().to("cpu").item()) if hasattr(freq, "detach") else float(freq)
        phase_f = (
            float(phase.detach().to("cpu").item()) if hasattr(phase, "detach") else float(phase)
        )
        bias_f = float(bias.detach().to("cpu").item()) if hasattr(bias, "detach") else float(bias)
        out = [
            math.sin(freq_f * math.log1p(abs(xv)) + phase_f) + bias_f for xv in x_list
        ]
        try:
            report(
                "neuron",
                "spiral_time_forward",
                {"freq": freq_f, "phase": phase_f, "bias": bias_f},
                "plugins",
            )
        except Exception:
            pass
        return out if len(out) != 1 else out[0]


__all__ = ["SpiralTimeNeuronPlugin"]

