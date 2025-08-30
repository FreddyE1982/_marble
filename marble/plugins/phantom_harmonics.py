from __future__ import annotations

"""Phantom harmonics neuron plugin.

Generates imaginary harmonics that overlay the input creating ghost-like
resonances intended to foster emergent oscillatory behaviour."""

from typing import Any, Tuple
import math

from ..wanderer import expose_learnable_params
from ..reporter import report


class PhantomHarmonicsNeuronPlugin:
    """Introduce phantom harmonic interference."""

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer: "Wanderer",
        *,
        phantom_scale: float = 1.0,
        phantom_shift: float = 0.0,
        phantom_bias: float = 0.0,
    ) -> Tuple[Any, Any, Any]:
        return (phantom_scale, phantom_shift, phantom_bias)

    def forward(self, neuron: "Neuron", input_value=None):
        x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)

        scale, shift, bias = 1.0, 0.0, 0.0
        wanderer = neuron._plugin_state.get("wanderer")
        if wanderer is not None:
            try:
                scale, shift, bias = self._params(wanderer)
            except Exception:
                pass

        torch = getattr(neuron, "_torch", None)
        if torch is not None and neuron._is_torch_tensor(x):
            phantom = torch.sin(x * scale + shift) * torch.cos(x * scale - shift)
            out = x + phantom + bias
            try:
                report(
                    "neuron",
                    "phantom_harmonics_forward",
                    {
                        "scale": float(scale.detach().to("cpu").item())
                        if hasattr(scale, "detach")
                        else float(scale),
                        "shift": float(shift.detach().to("cpu").item())
                        if hasattr(shift, "detach")
                        else float(shift),
                        "bias": float(bias.detach().to("cpu").item())
                        if hasattr(bias, "detach")
                        else float(bias),
                    },
                    "plugins",
                )
            except Exception:
                pass
            return out

        x_list = x if isinstance(x, list) else [float(x)]
        scale_f = float(scale.detach().to("cpu").item()) if hasattr(scale, "detach") else float(scale)
        shift_f = float(shift.detach().to("cpu").item()) if hasattr(shift, "detach") else float(shift)
        bias_f = float(bias.detach().to("cpu").item()) if hasattr(bias, "detach") else float(bias)
        out_vals = [xv + math.sin(xv * scale_f + shift_f) * math.cos(xv * scale_f - shift_f) + bias_f for xv in x_list]
        try:
            report(
                "neuron",
                "phantom_harmonics_forward",
                {"scale": scale_f, "shift": shift_f, "bias": bias_f},
                "plugins",
            )
        except Exception:
            pass
        return out_vals if len(out_vals) != 1 else out_vals[0]


__all__ = ["PhantomHarmonicsNeuronPlugin"]
