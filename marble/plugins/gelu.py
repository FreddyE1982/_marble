from __future__ import annotations

"""GELU neuron plugin with learnable scale."""

from typing import Any, Tuple
import math

from ..wanderer import expose_learnable_params
from ..reporter import report


class GELUNeuronPlugin:
    """Apply scaled Gaussian error linear unit."""

    @staticmethod
    @expose_learnable_params
    def _params(wanderer: "Wanderer", *, gelu_scale: float = 1.0) -> Tuple[Any]:
        return (gelu_scale,)

    def forward(self, neuron: "Neuron", input_value=None):
        x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)

        scale = 1.0
        wanderer = neuron._plugin_state.get("wanderer")
        if wanderer is not None:
            try:
                (scale,) = self._params(wanderer)
            except Exception:
                pass

        torch = getattr(neuron, "_torch", None)
        if torch is not None and neuron._is_torch_tensor(x):
            y = scale * 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
            try:
                report(
                    "neuron",
                    "gelu_forward",
                    {"scale": float(scale.detach().to("cpu").item()) if hasattr(scale, "detach") else float(scale)},
                    "plugins",
                )
            except Exception:
                pass
            return y

        x_list = x if isinstance(x, list) else [float(x)]
        scale_f = float(scale if not hasattr(scale, "detach") else scale.detach().to("cpu").item())
        out = [scale_f * 0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0))) for v in x_list]
        try:
            report("neuron", "gelu_forward", {"scale": scale_f}, "plugins")
        except Exception:
            pass
        return out if len(out) != 1 else out[0]


__all__ = ["GELUNeuronPlugin"]
