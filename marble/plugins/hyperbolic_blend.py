from __future__ import annotations

"""Hyperbolic blend neuron plugin.

Combines hyperbolic tangent and secant functions to create an activation
with both saturating and sharpening behaviour controlled by a learnable
``alpha`` parameter.
"""

from typing import Any, Tuple
import math

from ..wanderer import expose_learnable_params
from ..reporter import report


class HyperbolicBlendNeuronPlugin:
    """Mix tanh and sech responses via a learnable scale."""

    @staticmethod
    @expose_learnable_params
    def _params(wanderer: "Wanderer", *, hyper_alpha: float = 1.0) -> Tuple[Any]:
        return (hyper_alpha,)

    def forward(self, neuron: "Neuron", input_value=None):
        x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)

        alpha = 1.0
        wanderer = neuron._plugin_state.get("wanderer")
        if wanderer is not None:
            try:
                (alpha,) = self._params(wanderer)
            except Exception:
                pass

        torch = getattr(neuron, "_torch", None)
        if torch is not None and neuron._is_torch_tensor(x):
            y = torch.tanh(alpha * x) + torch.cosh(x).reciprocal()
            try:
                report(
                    "neuron",
                    "hyperbolic_blend_forward",
                    {"alpha": float(alpha.detach().to("cpu").item()) if hasattr(alpha, "detach") else float(alpha)},
                    "plugins",
                )
            except Exception:
                pass
            return y

        x_list = x if isinstance(x, list) else [float(x)]
        alpha_f = float(alpha.detach().to("cpu").item()) if hasattr(alpha, "detach") else float(alpha)
        out = [math.tanh(alpha_f * v) + 1.0 / math.cosh(v) for v in x_list]
        try:
            report("neuron", "hyperbolic_blend_forward", {"alpha": alpha_f}, "plugins")
        except Exception:
            pass
        return out if len(out) != 1 else out[0]


__all__ = ["HyperbolicBlendNeuronPlugin"]

