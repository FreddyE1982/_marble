from __future__ import annotations

"""SoftPlus neuron plugin with learnable beta and threshold."""

from typing import Any, Tuple
import math

from ..wanderer import expose_learnable_params
from ..reporter import report


class SoftPlusNeuronPlugin:
    """Apply SoftPlus activation with learnable parameters."""

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer: "Wanderer", *, splus_beta: float = 1.0, splus_threshold: float = 20.0
    ) -> Tuple[Any, Any]:
        return splus_beta, splus_threshold

    def forward(self, neuron: "Neuron", input_value=None):
        x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)

        beta, threshold = 1.0, 20.0
        wanderer = neuron._plugin_state.get("wanderer")
        if wanderer is not None:
            try:
                beta, threshold = self._params(wanderer)
            except Exception:
                pass

        torch = getattr(neuron, "_torch", None)
        if torch is not None and neuron._is_torch_tensor(x):
            beta_t = torch.as_tensor(beta, device=x.device, dtype=x.dtype)
            thresh_t = torch.as_tensor(threshold, device=x.device, dtype=x.dtype)
            y = torch.where(
                beta_t * x < thresh_t,
                torch.log1p(torch.exp(beta_t * x)) / beta_t,
                x,
            )
            try:
                report(
                    "neuron",
                    "softplus_forward",
                    {
                        "beta": float(beta_t.detach().to("cpu").item()),
                        "threshold": float(thresh_t.detach().to("cpu").item()),
                    },
                    "plugins",
                )
            except Exception:
                pass
            return y

        x_list = x if isinstance(x, list) else [float(x)]
        beta_f = float(beta if not hasattr(beta, "detach") else beta.detach().to("cpu").item())
        thresh_f = float(threshold if not hasattr(threshold, "detach") else threshold.detach().to("cpu").item())
        out = [
            (1.0 / beta_f) * math.log1p(math.exp(beta_f * v)) if beta_f * v < thresh_f else v
            for v in x_list
        ]
        try:
            report(
                "neuron",
                "softplus_forward",
                {"beta": beta_f, "threshold": thresh_f},
                "plugins",
            )
        except Exception:
            pass
        return out if len(out) != 1 else out[0]


__all__ = ["SoftPlusNeuronPlugin"]
