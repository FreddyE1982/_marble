from __future__ import annotations

"""Fractal logistic neuron plugin.

Iteratively applies the logistic map to the input, creating fractal-like
behaviour controlled by learnable ``r`` and iteration count parameters.
"""

from typing import Any, Tuple
import math

from ..wanderer import expose_learnable_params
from ..reporter import report


class FractalLogisticNeuronPlugin:
    """Logistic map activation with learnable chaos parameters."""

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer: "Wanderer", *, fractal_r: float = 3.5, fractal_iters: float = 3.0
    ) -> Tuple[Any, Any]:
        return (fractal_r, fractal_iters)

    def forward(self, neuron: "Neuron", input_value=None):
        x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)

        r = 3.5
        iters = 3.0
        wanderer = neuron._plugin_state.get("wanderer")
        if wanderer is not None:
            try:
                r, iters = self._params(wanderer)
            except Exception:
                pass

        iters_i = int(
            iters.detach().to("cpu").item() if hasattr(iters, "detach") else iters
        )

        torch = getattr(neuron, "_torch", None)
        if torch is not None and neuron._is_torch_tensor(x):
            v = torch.sigmoid(x)
            for _ in range(max(0, iters_i)):
                v = r * v * (1 - v)
            try:
                report(
                    "neuron",
                    "fractal_logistic_forward",
                    {
                        "r": float(r.detach().to("cpu").item()) if hasattr(r, "detach") else float(r),
                        "iters": iters_i,
                    },
                    "plugins",
                )
            except Exception:
                pass
            return v

        x_list = x if isinstance(x, list) else [float(x)]
        r_f = float(r.detach().to("cpu").item()) if hasattr(r, "detach") else float(r)
        out = []
        for xv in x_list:
            v = 1.0 / (1.0 + math.exp(-xv))
            for _ in range(max(0, iters_i)):
                v = r_f * v * (1.0 - v)
            out.append(v)
        try:
            report(
                "neuron",
                "fractal_logistic_forward",
                {"r": r_f, "iters": iters_i},
                "plugins",
            )
        except Exception:
            pass
        return out if len(out) != 1 else out[0]


__all__ = ["FractalLogisticNeuronPlugin"]

