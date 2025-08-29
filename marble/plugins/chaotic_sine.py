from __future__ import annotations

"""Chaotic sine neuron plugin.

Iterates a logistic map and feeds the result through a sine function to
create a simple chaotic oscillation. All parameters are exposed through
``expose_learnable_params`` so the wanderer can tune the behaviour.
"""

from typing import Any, Tuple
import math

from ..wanderer import expose_learnable_params
from ..reporter import report


class ChaoticSineNeuronPlugin:
    """Apply logistic-map driven sine activation."""

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer: "Wanderer",
        *,
        chaos_r: float = 3.7,
        chaos_iters: int = 2,
        chaos_bias: float = 0.0,
    ) -> Tuple[Any, Any, Any]:
        return (chaos_r, chaos_iters, chaos_bias)

    def forward(self, neuron: "Neuron", input_value=None):
        x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)

        r = 3.7
        iters = 2
        bias = 0.0
        wanderer = neuron._plugin_state.get("wanderer")
        if wanderer is not None:
            try:
                r, iters, bias = self._params(wanderer)
            except Exception:
                pass

        torch = getattr(neuron, "_torch", None)
        if torch is not None and neuron._is_torch_tensor(x):
            v = torch.sigmoid(x)
            for _ in range(int(iters)):
                v = r * v * (1 - v)
            y = torch.sin(2 * math.pi * v) + bias
            try:
                report(
                    "neuron",
                    "chaotic_sine_forward",
                    {
                        "r": float(r.detach().to("cpu").item())
                        if hasattr(r, "detach")
                        else float(r),
                        "iters": int(iters),
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
        r_f = float(r.detach().to("cpu").item()) if hasattr(r, "detach") else float(r)
        bias_f = (
            float(bias.detach().to("cpu").item()) if hasattr(bias, "detach") else float(bias)
        )
        out = []
        for xv in x_list:
            v = 1.0 / (1.0 + math.exp(-xv))
            for _ in range(int(iters)):
                v = r_f * v * (1.0 - v)
            out.append(math.sin(2 * math.pi * v) + bias_f)
        try:
            report(
                "neuron",
                "chaotic_sine_forward",
                {"r": r_f, "iters": int(iters), "bias": bias_f},
                "plugins",
            )
        except Exception:
            pass
        return out if len(out) != 1 else out[0]


__all__ = ["ChaoticSineNeuronPlugin"]

