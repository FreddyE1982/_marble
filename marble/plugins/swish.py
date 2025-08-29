from __future__ import annotations

"""Swish neuron plugin applying x * sigmoid(beta*x)."""

from typing import Any, Tuple
import math

from ..wanderer import expose_learnable_params
from ..reporter import report


class SwishNeuronPlugin:
    """Apply Swish activation with learnable beta."""

    @staticmethod
    @expose_learnable_params
    def _params(wanderer: "Wanderer", *, swish_beta: float = 1.0) -> Tuple[Any]:
        return (swish_beta,)

    def forward(self, neuron: "Neuron", input_value=None):
        x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)

        beta = 1.0
        wanderer = neuron._plugin_state.get("wanderer")
        if wanderer is not None:
            try:
                (beta,) = self._params(wanderer)
            except Exception:
                pass

        torch = getattr(neuron, "_torch", None)
        if torch is not None and neuron._is_torch_tensor(x):
            y = x * torch.sigmoid(beta * x)
            try:
                report(
                    "neuron",
                    "swish_forward",
                    {"beta": float(beta.detach().to("cpu").item()) if hasattr(beta, "detach") else float(beta)},
                    "plugins",
                )
            except Exception:
                pass
            return y

        x_list = x if isinstance(x, list) else [float(x)]
        beta_f = float(beta if not hasattr(beta, "detach") else beta.detach().to("cpu").item())
        out = [v * (1.0 / (1.0 + math.exp(-beta_f * v))) for v in x_list]
        try:
            report("neuron", "swish_forward", {"beta": beta_f}, "plugins")
        except Exception:
            pass
        return out if len(out) != 1 else out[0]


__all__ = ["SwishNeuronPlugin"]
