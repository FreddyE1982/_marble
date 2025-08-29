from __future__ import annotations

"""Fourier series neuron plugin.

This plugin combines sine and cosine harmonics with learnable amplitudes,
frequencies and phases. All parameters are exposed via
``expose_learnable_params`` so optimisation frameworks can tune each
harmonic individually.
"""

from typing import Any, Tuple

from ..wanderer import expose_learnable_params


class FourierSeriesNeuronPlugin:
    """Apply a small Fourier series to the neuron value."""

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer: "Wanderer",
        *,
        fourier_a1: float = 1.0,
        fourier_f1: float = 1.0,
        fourier_p1: float = 0.0,
        fourier_a2: float = 0.5,
        fourier_f2: float = 2.0,
        fourier_p2: float = 0.0,
        fourier_bias: float = 0.0,
    ) -> Tuple[Any, Any, Any, Any, Any, Any, Any]:
        return (
            fourier_a1,
            fourier_f1,
            fourier_p1,
            fourier_a2,
            fourier_f2,
            fourier_p2,
            fourier_bias,
        )

    def forward(self, neuron: "Neuron", input_value=None):
        x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)

        a1, f1, p1 = 1.0, 1.0, 0.0
        a2, f2, p2 = 0.5, 2.0, 0.0
        bias = 0.0
        wanderer = neuron._plugin_state.get("wanderer")
        if wanderer is not None:
            try:
                a1, f1, p1, a2, f2, p2, bias = self._params(wanderer)
            except Exception:
                pass

        torch = getattr(neuron, "_torch", None)
        if torch is not None and neuron._is_torch_tensor(x):
            y = (
                a1 * torch.sin(f1 * x + p1)
                + a2 * torch.cos(f2 * x + p2)
                + bias
            )
            return y

        x_list = x if isinstance(x, list) else [float(x)]
        a1_f = float(a1 if not hasattr(a1, "detach") else a1.detach().to("cpu").item())
        f1_f = float(f1 if not hasattr(f1, "detach") else f1.detach().to("cpu").item())
        p1_f = float(p1 if not hasattr(p1, "detach") else p1.detach().to("cpu").item())
        a2_f = float(a2 if not hasattr(a2, "detach") else a2.detach().to("cpu").item())
        f2_f = float(f2 if not hasattr(f2, "detach") else f2.detach().to("cpu").item())
        p2_f = float(p2 if not hasattr(p2, "detach") else p2.detach().to("cpu").item())
        b_f = float(bias if not hasattr(bias, "detach") else bias.detach().to("cpu").item())
        out = [
            a1_f * __import__("math").sin(f1_f * v + p1_f)
            + a2_f * __import__("math").cos(f2_f * v + p2_f)
            + b_f
            for v in map(float, x_list)
        ]
        return out if len(out) != 1 else out[0]


__all__ = ["FourierSeriesNeuronPlugin"]

