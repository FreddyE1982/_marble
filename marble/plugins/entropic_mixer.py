from __future__ import annotations

"""Entropic mixer neuron plugin.

Combines exponential decay with cosine interference to sculpt the entropy
of activations. All parameters are wanderer-learnable.
"""

from typing import Any, Tuple
import math

from ..wanderer import expose_learnable_params
from ..reporter import report


class EntropicMixerNeuronPlugin:
    """Decay input energy and apply cosine mixing."""

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer: "Wanderer",
        *,
        ent_alpha: float = 1.0,
        ent_beta: float = 1.0,
        ent_bias: float = 0.0,
    ) -> Tuple[Any, Any, Any]:
        return (ent_alpha, ent_beta, ent_bias)

    def forward(self, neuron: "Neuron", input_value=None):
        x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)

        alpha = 1.0
        beta = 1.0
        bias = 0.0
        wanderer = neuron._plugin_state.get("wanderer")
        if wanderer is not None:
            try:
                alpha, beta, bias = self._params(wanderer)
            except Exception:
                pass

        torch = getattr(neuron, "_torch", None)
        if torch is not None and neuron._is_torch_tensor(x):
            y = torch.exp(-alpha * x * x) * torch.cos(beta * x) * x + bias
            try:
                report(
                    "neuron",
                    "entropic_mixer_forward",
                    {
                        "alpha": float(alpha.detach().to("cpu").item())
                        if hasattr(alpha, "detach")
                        else float(alpha),
                        "beta": float(beta.detach().to("cpu").item())
                        if hasattr(beta, "detach")
                        else float(beta),
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
        alpha_f = (
            float(alpha.detach().to("cpu").item()) if hasattr(alpha, "detach") else float(alpha)
        )
        beta_f = float(beta.detach().to("cpu").item()) if hasattr(beta, "detach") else float(beta)
        bias_f = float(bias.detach().to("cpu").item()) if hasattr(bias, "detach") else float(bias)
        out = [
            math.exp(-alpha_f * xv * xv) * math.cos(beta_f * xv) * xv + bias_f for xv in x_list
        ]
        try:
            report(
                "neuron",
                "entropic_mixer_forward",
                {"alpha": alpha_f, "beta": beta_f, "bias": bias_f},
                "plugins",
            )
        except Exception:
            pass
        return out if len(out) != 1 else out[0]


__all__ = ["EntropicMixerNeuronPlugin"]

