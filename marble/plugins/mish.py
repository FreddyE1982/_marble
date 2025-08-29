from __future__ import annotations

"""Mish neuron plugin applying x * tanh(softplus(beta*x))."""

from typing import Any, Tuple
import math

from ..wanderer import expose_learnable_params
from ..reporter import report


class MishNeuronPlugin:
    """Apply Mish activation with learnable beta."""

    @staticmethod
    @expose_learnable_params
    def _params(wanderer: "Wanderer", *, mish_beta: float = 1.0) -> Tuple[Any]:
        return (mish_beta,)

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
            y = x * torch.tanh(torch.nn.functional.softplus(beta * x))
            try:
                report(
                    "neuron",
                    "mish_forward",
                    {"beta": float(beta.detach().to("cpu").item()) if hasattr(beta, "detach") else float(beta)},
                    "plugins",
                )
            except Exception:
                pass
            return y

        x_list = x if isinstance(x, list) else [float(x)]
        beta_f = float(beta if not hasattr(beta, "detach") else beta.detach().to("cpu").item())
        out = [v * math.tanh(math.log1p(math.exp(beta_f * v))) for v in x_list]
        try:
            report("neuron", "mish_forward", {"beta": beta_f}, "plugins")
        except Exception:
            pass
        return out if len(out) != 1 else out[0]


__all__ = ["MishNeuronPlugin"]
