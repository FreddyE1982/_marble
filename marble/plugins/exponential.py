from __future__ import annotations

"""Exponential neuron plugin with learnable rate, scale and bias.

Implements ``y = scale * exp(rate * x) + bias`` with all parameters exposed
through :func:`expose_learnable_params`.
"""

from typing import Any, Tuple
import math

from ..wanderer import expose_learnable_params
from ..reporter import report


class ExponentialNeuronPlugin:
    """Apply a learnable exponential transform to the neuron value."""

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer: "Wanderer",
        *,
        exp_rate: float = 1.0,
        exp_scale: float = 1.0,
        exp_bias: float = 0.0,
    ) -> Tuple[Any, Any, Any]:
        return exp_rate, exp_scale, exp_bias

    def forward(self, neuron: "Neuron", input_value=None):
        x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)
        rate, scale, bias = 1.0, 1.0, 0.0
        wanderer = neuron._plugin_state.get("wanderer")
        if wanderer is not None:
            try:
                rate, scale, bias = self._params(wanderer)
            except Exception:
                pass

        torch = getattr(neuron, "_torch", None)
        if torch is not None and neuron._is_torch_tensor(x):
            y = scale * torch.exp(rate * x) + bias
            try:
                report(
                    "neuron",
                    "exp_forward",
                    {"rate": float(rate.detach().to("cpu").item()) if hasattr(rate, "detach") else float(rate)},
                    "plugins",
                )
            except Exception:
                pass
            return y

        x_list = x if isinstance(x, list) else [float(x)]
        rate_f = float(rate if not hasattr(rate, "detach") else rate.detach().to("cpu").item())
        scale_f = float(scale if not hasattr(scale, "detach") else scale.detach().to("cpu").item())
        bias_f = float(bias if not hasattr(bias, "detach") else bias.detach().to("cpu").item())
        out = [scale_f * math.exp(rate_f * float(v)) + bias_f for v in x_list]
        try:
            report("neuron", "exp_forward", {"rate": rate_f}, "plugins")
        except Exception:
            pass
        return out if len(out) != 1 else out[0]


__all__ = ["ExponentialNeuronPlugin"]

