from __future__ import annotations

"""Lattice resonance neuron plugin.

Projects the input onto a modular lattice and scales the result. Useful for
creating repeating patterns with learnable period and scale.
"""

from typing import Any, Tuple
import math

from ..wanderer import expose_learnable_params
from ..reporter import report


class LatticeResonanceNeuronPlugin:
    """Apply modular reduction followed by scaling and bias."""

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer: "Wanderer",
        *,
        lattice_mod: float = 2.0,
        lattice_scale: float = 1.0,
        lattice_bias: float = 0.0,
    ) -> Tuple[Any, Any, Any]:
        return (lattice_mod, lattice_scale, lattice_bias)

    def forward(self, neuron: "Neuron", input_value=None):
        x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)

        mod = 2.0
        scale = 1.0
        bias = 0.0
        wanderer = neuron._plugin_state.get("wanderer")
        if wanderer is not None:
            try:
                mod, scale, bias = self._params(wanderer)
            except Exception:
                pass

        torch = getattr(neuron, "_torch", None)
        if torch is not None and neuron._is_torch_tensor(x):
            v = torch.remainder(x, mod) / mod
            y = v * scale + bias
            try:
                report(
                    "neuron",
                    "lattice_resonance_forward",
                    {
                        "mod": float(mod.detach().to("cpu").item())
                        if hasattr(mod, "detach")
                        else float(mod),
                        "scale": float(scale.detach().to("cpu").item())
                        if hasattr(scale, "detach")
                        else float(scale),
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
        mod_f = float(mod.detach().to("cpu").item()) if hasattr(mod, "detach") else float(mod)
        scale_f = (
            float(scale.detach().to("cpu").item()) if hasattr(scale, "detach") else float(scale)
        )
        bias_f = float(bias.detach().to("cpu").item()) if hasattr(bias, "detach") else float(bias)
        out = [math.fmod(xv, mod_f) / mod_f * scale_f + bias_f for xv in x_list]
        try:
            report(
                "neuron",
                "lattice_resonance_forward",
                {"mod": mod_f, "scale": scale_f, "bias": bias_f},
                "plugins",
            )
        except Exception:
            pass
        return out if len(out) != 1 else out[0]


__all__ = ["LatticeResonanceNeuronPlugin"]

