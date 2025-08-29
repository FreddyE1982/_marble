from __future__ import annotations

"""Leaky exponential linear unit with learnable alpha and beta."""

from typing import Any, Tuple
import math

from ..wanderer import expose_learnable_params
from ..reporter import report


class LeakyExpNeuronPlugin:
    """Like ELU but with learnable alpha and exponential rate."""

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer: "Wanderer", *, leak_alpha: float = 1.0, leak_beta: float = 1.0
    ) -> Tuple[Any, Any]:
        return leak_alpha, leak_beta

    def forward(self, neuron: "Neuron", input_value=None):
        x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)

        alpha, beta = 1.0, 1.0
        wanderer = neuron._plugin_state.get("wanderer")
        if wanderer is not None:
            try:
                alpha, beta = self._params(wanderer)
            except Exception:
                pass

        torch = getattr(neuron, "_torch", None)
        if torch is not None and neuron._is_torch_tensor(x):
            pos = torch.clamp(x, min=0.0)
            neg = torch.clamp(x, max=0.0)
            y = pos + alpha * (torch.exp(beta * neg) - 1.0)
            try:
                report(
                    "neuron",
                    "leakyexp_forward",
                    {
                        "alpha": float(alpha.detach().to("cpu").item()) if hasattr(alpha, "detach") else float(alpha),
                        "beta": float(beta.detach().to("cpu").item()) if hasattr(beta, "detach") else float(beta),
                    },
                    "plugins",
                )
            except Exception:
                pass
            return y

        x_list = x if isinstance(x, list) else [float(x)]
        alpha_f = float(alpha if not hasattr(alpha, "detach") else alpha.detach().to("cpu").item())
        beta_f = float(beta if not hasattr(beta, "detach") else beta.detach().to("cpu").item())
        out = [v if v > 0 else alpha_f * (math.exp(beta_f * v) - 1.0) for v in x_list]
        try:
            report("neuron", "leakyexp_forward", {"alpha": alpha_f, "beta": beta_f}, "plugins")
        except Exception:
            pass
        return out if len(out) != 1 else out[0]


__all__ = ["LeakyExpNeuronPlugin"]
