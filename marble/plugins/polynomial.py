from __future__ import annotations

"""Polynomial neuron plugin with learnable quadratic coefficients.

The plugin evaluates ``y = a * x**2 + b * x + c`` with parameters ``a``, ``b``
and ``c`` exposed via :func:`expose_learnable_params`.
"""

from typing import Any, Tuple

from ..wanderer import expose_learnable_params
from ..reporter import report


class PolynomialNeuronPlugin:
    """Apply a learnable quadratic polynomial to the neuron value."""

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer: "Wanderer",
        *,
        poly_a: float = 1.0,
        poly_b: float = 0.0,
        poly_c: float = 0.0,
    ) -> Tuple[Any, Any, Any]:
        return poly_a, poly_b, poly_c

    def forward(self, neuron: "Neuron", input_value=None):
        x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)

        a, b, c = 1.0, 0.0, 0.0
        wanderer = neuron._plugin_state.get("wanderer")
        if wanderer is not None:
            try:
                a, b, c = self._params(wanderer)
            except Exception:
                pass

        torch = getattr(neuron, "_torch", None)
        if torch is not None and neuron._is_torch_tensor(x):
            y = a * (x ** 2) + b * x + c
            try:
                report(
                    "neuron",
                    "poly_forward",
                    {"a": float(a.detach().to("cpu").item()) if hasattr(a, "detach") else float(a)},
                    "plugins",
                )
            except Exception:
                pass
            return y

        x_list = x if isinstance(x, list) else [float(x)]
        a_f = float(a if not hasattr(a, "detach") else a.detach().to("cpu").item())
        b_f = float(b if not hasattr(b, "detach") else b.detach().to("cpu").item())
        c_f = float(c if not hasattr(c, "detach") else c.detach().to("cpu").item())
        out = [a_f * float(v) * float(v) + b_f * float(v) + c_f for v in x_list]
        try:
            report("neuron", "poly_forward", {"a": a_f}, "plugins")
        except Exception:
            pass
        return out if len(out) != 1 else out[0]


__all__ = ["PolynomialNeuronPlugin"]

