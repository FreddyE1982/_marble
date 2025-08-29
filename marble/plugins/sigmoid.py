from __future__ import annotations

"""Parameterized sigmoid neuron plugin.

Implements a logistic function with learnable scale, slope, midpoint and
bias, all registered via ``expose_learnable_params``.
"""

from typing import Any, Tuple

from ..wanderer import expose_learnable_params


class SigmoidNeuronPlugin:
    """Apply ``scale / (1 + exp(-k*(x - x0))) + bias``."""

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer: "Wanderer",
        *,
        sig_scale: float = 1.0,
        sig_k: float = 1.0,
        sig_x0: float = 0.0,
        sig_bias: float = 0.0,
    ) -> Tuple[Any, Any, Any, Any]:
        return sig_scale, sig_k, sig_x0, sig_bias

    def forward(self, neuron: "Neuron", input_value=None):
        x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)

        scale, k, x0, bias = 1.0, 1.0, 0.0, 0.0
        wanderer = neuron._plugin_state.get("wanderer")
        if wanderer is not None:
            try:
                scale, k, x0, bias = self._params(wanderer)
            except Exception:
                pass

        torch = getattr(neuron, "_torch", None)
        if torch is not None and neuron._is_torch_tensor(x):
            y = scale / (1 + torch.exp(-k * (x - x0))) + bias
            return y

        import math

        x_list = x if isinstance(x, list) else [float(x)]
        scale_f = float(scale if not hasattr(scale, "detach") else scale.detach().to("cpu").item())
        k_f = float(k if not hasattr(k, "detach") else k.detach().to("cpu").item())
        x0_f = float(x0 if not hasattr(x0, "detach") else x0.detach().to("cpu").item())
        bias_f = float(bias if not hasattr(bias, "detach") else bias.detach().to("cpu").item())
        out = [scale_f / (1 + math.exp(-k_f * (v - x0_f))) + bias_f for v in map(float, x_list)]
        return out if len(out) != 1 else out[0]


__all__ = ["SigmoidNeuronPlugin"]

