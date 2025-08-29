from __future__ import annotations

"""Gaussian neuron plugin with learnable mean, sigma, scale and bias.

The plugin evaluates a Gaussian radial basis function for the neuron input
using parameters exposed via :func:`expose_learnable_params`:

``y = scale * exp(-((x - mean)**2) / (2 * sigma**2)) + bias``
"""

from typing import Any, Tuple
import math

from ..wanderer import expose_learnable_params
from ..reporter import report


class GaussianNeuronPlugin:
    """Apply a learnable Gaussian RBF to the neuron value."""

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer: "Wanderer",
        *,
        gauss_mean: float = 0.0,
        gauss_sigma: float = 1.0,
        gauss_scale: float = 1.0,
        gauss_bias: float = 0.0,
    ) -> Tuple[Any, Any, Any, Any]:
        return gauss_mean, gauss_sigma, gauss_scale, gauss_bias

    def forward(self, neuron: "Neuron", input_value=None):
        x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)
        mean, sigma, scale, bias = 0.0, 1.0, 1.0, 0.0
        wanderer = neuron._plugin_state.get("wanderer")
        if wanderer is not None:
            try:
                mean, sigma, scale, bias = self._params(wanderer)
            except Exception:
                pass

        torch = getattr(neuron, "_torch", None)
        if torch is not None and neuron._is_torch_tensor(x):
            denom = 2.0 * sigma * sigma
            y = scale * torch.exp(-((x - mean) ** 2) / denom) + bias
            try:
                report(
                    "neuron",
                    "gaussian_forward",
                    {"sigma": float(sigma.detach().to("cpu").item()) if hasattr(sigma, "detach") else float(sigma)},
                    "plugins",
                )
            except Exception:
                pass
            return y

        x_list = x if isinstance(x, list) else [float(x)]
        mean_f = float(mean if not hasattr(mean, "detach") else mean.detach().to("cpu").item())
        sigma_f = float(sigma if not hasattr(sigma, "detach") else sigma.detach().to("cpu").item())
        scale_f = float(scale if not hasattr(scale, "detach") else scale.detach().to("cpu").item())
        bias_f = float(bias if not hasattr(bias, "detach") else bias.detach().to("cpu").item())
        denom_f = 2.0 * sigma_f * sigma_f if sigma_f != 0 else 1.0
        out = [scale_f * math.exp(-((float(v) - mean_f) ** 2) / denom_f) + bias_f for v in x_list]
        try:
            report("neuron", "gaussian_forward", {"sigma": sigma_f}, "plugins")
        except Exception:
            pass
        return out if len(out) != 1 else out[0]


__all__ = ["GaussianNeuronPlugin"]

