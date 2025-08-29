from __future__ import annotations

"""Radial basis function neuron plugin with learnable center, gamma, scale and bias.

Computes ``y = scale * exp(-gamma * (x - center)**2) + bias`` exposing all
parameters via :func:`expose_learnable_params`.
"""

from typing import Any, Tuple
import math

from ..wanderer import expose_learnable_params
from ..reporter import report


class RBFNeuronPlugin:
    """Apply a learnable radial basis function to the neuron value."""

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer: "Wanderer",
        *,
        rbf_center: float = 0.0,
        rbf_gamma: float = 1.0,
        rbf_scale: float = 1.0,
        rbf_bias: float = 0.0,
    ) -> Tuple[Any, Any, Any, Any]:
        return rbf_center, rbf_gamma, rbf_scale, rbf_bias

    def forward(self, neuron: "Neuron", input_value=None):
        x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)
        center, gamma, scale, bias = 0.0, 1.0, 1.0, 0.0
        wanderer = neuron._plugin_state.get("wanderer")
        if wanderer is not None:
            try:
                center, gamma, scale, bias = self._params(wanderer)
            except Exception:
                pass

        torch = getattr(neuron, "_torch", None)
        if torch is not None and neuron._is_torch_tensor(x):
            y = scale * torch.exp(-gamma * (x - center) ** 2) + bias
            try:
                report(
                    "neuron",
                    "rbf_forward",
                    {"gamma": float(gamma.detach().to("cpu").item()) if hasattr(gamma, "detach") else float(gamma)},
                    "plugins",
                )
            except Exception:
                pass
            return y

        x_list = x if isinstance(x, list) else [float(x)]
        center_f = float(center if not hasattr(center, "detach") else center.detach().to("cpu").item())
        gamma_f = float(gamma if not hasattr(gamma, "detach") else gamma.detach().to("cpu").item())
        scale_f = float(scale if not hasattr(scale, "detach") else scale.detach().to("cpu").item())
        bias_f = float(bias if not hasattr(bias, "detach") else bias.detach().to("cpu").item())
        out = [scale_f * math.exp(-gamma_f * (float(v) - center_f) ** 2) + bias_f for v in x_list]
        try:
            report("neuron", "rbf_forward", {"gamma": gamma_f}, "plugins")
        except Exception:
            pass
        return out if len(out) != 1 else out[0]


__all__ = ["RBFNeuronPlugin"]

