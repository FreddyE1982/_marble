from __future__ import annotations

"""Quantum mirror neuron plugin.

Reflects inputs through an imaginary quantum mirror where amplitude and phase
interfere in a non-classical fashion. Designed to spark emergent behaviour
through self-interference of signals."""

from typing import Any, Tuple
import math

from ..wanderer import expose_learnable_params
from ..reporter import report


class QuantumMirrorNeuronPlugin:
    """Apply mirror-like quantum interference."""

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer: "Wanderer",
        *,
        mirror_coeff: float = 1.0,
        mirror_bias: float = 0.0,
    ) -> Tuple[Any, Any]:
        return (mirror_coeff, mirror_bias)

    def forward(self, neuron: "Neuron", input_value=None):
        x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)

        coeff, bias = 1.0, 0.0
        wanderer = neuron._plugin_state.get("wanderer")
        if wanderer is not None:
            try:
                coeff, bias = self._params(wanderer)
            except Exception:
                pass

        torch = getattr(neuron, "_torch", None)
        if torch is not None and neuron._is_torch_tensor(x):
            out = x * torch.cos(coeff * x) - x * torch.sin(coeff * x) + bias
            try:
                report(
                    "neuron",
                    "quantum_mirror_forward",
                    {
                        "coeff": float(coeff.detach().to("cpu").item())
                        if hasattr(coeff, "detach")
                        else float(coeff),
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
        coeff_f = float(coeff.detach().to("cpu").item()) if hasattr(coeff, "detach") else float(coeff)
        bias_f = float(bias.detach().to("cpu").item()) if hasattr(bias, "detach") else float(bias)
        out_vals = [xi * math.cos(coeff_f * xi) - xi * math.sin(coeff_f * xi) + bias_f for xi in x_list]
        try:
            report("neuron", "quantum_mirror_forward", {"coeff": coeff_f, "bias": bias_f}, "plugins")
        except Exception:
            pass
        return out_vals if len(out_vals) != 1 else out_vals[0]


__all__ = ["QuantumMirrorNeuronPlugin"]
