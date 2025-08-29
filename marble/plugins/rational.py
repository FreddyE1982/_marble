from __future__ import annotations

"""Rational function neuron plugin.

Computes a ratio of two linear polynomials with learnable coefficients.
All coefficients are exposed via ``expose_learnable_params`` and a bias
term is added to retain expressive power.
"""

from typing import Any, Tuple

from ..wanderer import expose_learnable_params


class RationalNeuronPlugin:
    """Apply a rational function ``(a1*x + b1)/(a2*x + b2) + bias``."""

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer: "Wanderer",
        *,
        rat_a1: float = 1.0,
        rat_b1: float = 0.0,
        rat_a2: float = 1.0,
        rat_b2: float = 1.0,
        rat_bias: float = 0.0,
    ) -> Tuple[Any, Any, Any, Any, Any]:
        return rat_a1, rat_b1, rat_a2, rat_b2, rat_bias

    def forward(self, neuron: "Neuron", input_value=None):
        x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)

        a1, b1, a2, b2, bias = 1.0, 0.0, 1.0, 1.0, 0.0
        wanderer = neuron._plugin_state.get("wanderer")
        if wanderer is not None:
            try:
                a1, b1, a2, b2, bias = self._params(wanderer)
            except Exception:
                pass

        torch = getattr(neuron, "_torch", None)
        eps = 1e-6
        if torch is not None and neuron._is_torch_tensor(x):
            denom = a2 * x + b2
            y = (a1 * x + b1) / (denom + eps) + bias
            return y

        x_list = x if isinstance(x, list) else [float(x)]
        a1_f = float(a1 if not hasattr(a1, "detach") else a1.detach().to("cpu").item())
        b1_f = float(b1 if not hasattr(b1, "detach") else b1.detach().to("cpu").item())
        a2_f = float(a2 if not hasattr(a2, "detach") else a2.detach().to("cpu").item())
        b2_f = float(b2 if not hasattr(b2, "detach") else b2.detach().to("cpu").item())
        bias_f = float(bias if not hasattr(bias, "detach") else bias.detach().to("cpu").item())
        out = [
            (a1_f * v + b1_f) / (a2_f * v + b2_f + eps) + bias_f for v in map(float, x_list)
        ]
        return out if len(out) != 1 else out[0]


__all__ = ["RationalNeuronPlugin"]

