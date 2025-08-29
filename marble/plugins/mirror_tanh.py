from __future__ import annotations

"""Mirror tanh neuron plugin.

Contrasts the response of ``tanh(x)`` with ``tanh(|x|)`` to emphasise
sign-sensitive deviations. Scale and bias are learnable parameters.
"""

from typing import Any, Tuple
import math

from ..wanderer import expose_learnable_params
from ..reporter import report


class MirrorTanhNeuronPlugin:
    """Highlight differences between positive and negative activations."""

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer: "Wanderer", *, mirror_scale: float = 1.0, mirror_bias: float = 0.0
    ) -> Tuple[Any, Any]:
        return (mirror_scale, mirror_bias)

    def forward(self, neuron: "Neuron", input_value=None):
        x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)

        scale = 1.0
        bias = 0.0
        wanderer = neuron._plugin_state.get("wanderer")
        if wanderer is not None:
            try:
                scale, bias = self._params(wanderer)
            except Exception:
                pass

        torch = getattr(neuron, "_torch", None)
        if torch is not None and neuron._is_torch_tensor(x):
            y = scale * (torch.tanh(x) - torch.tanh(torch.abs(x))) + bias
            try:
                report(
                    "neuron",
                    "mirror_tanh_forward",
                    {
                        "scale": float(scale.detach().to("cpu").item())
                        if hasattr(scale, "detach")
                        else float(scale),
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
        scale_f = (
            float(scale.detach().to("cpu").item()) if hasattr(scale, "detach") else float(scale)
        )
        bias_f = float(bias.detach().to("cpu").item()) if hasattr(bias, "detach") else float(bias)
        out = [
            scale_f * (math.tanh(xv) - math.tanh(abs(xv))) + bias_f for xv in x_list
        ]
        try:
            report(
                "neuron",
                "mirror_tanh_forward",
                {"scale": scale_f, "bias": bias_f},
                "plugins",
            )
        except Exception:
            pass
        return out if len(out) != 1 else out[0]


__all__ = ["MirrorTanhNeuronPlugin"]

