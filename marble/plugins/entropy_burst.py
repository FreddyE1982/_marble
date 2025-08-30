from __future__ import annotations

"""Entropy burst neuron plugin.

Amplifies signals logarithmically based on magnitude and injects a bias burst
representing spontaneous entropy fluctuations."""

from typing import Any, Tuple
import math

from ..wanderer import expose_learnable_params
from ..reporter import report


class EntropyBurstNeuronPlugin:
    """Create entropy-driven activation bursts."""

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer: "Wanderer",
        *,
        burst_intensity: float = 1.0,
        burst_bias: float = 0.0,
    ) -> Tuple[Any, Any]:
        return (burst_intensity, burst_bias)

    def forward(self, neuron: "Neuron", input_value=None):
        x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)

        inten, bias = 1.0, 0.0
        wanderer = neuron._plugin_state.get("wanderer")
        if wanderer is not None:
            try:
                inten, bias = self._params(wanderer)
            except Exception:
                pass

        torch = getattr(neuron, "_torch", None)
        if torch is not None and neuron._is_torch_tensor(x):
            out = torch.sign(x) * torch.log1p(torch.abs(x) * inten) + bias
            try:
                report(
                    "neuron",
                    "entropy_burst_forward",
                    {
                        "intensity": float(inten.detach().to("cpu").item())
                        if hasattr(inten, "detach")
                        else float(inten),
                        "bias": float(bias.detach().to("cpu").item())
                        if hasattr(bias, "detach")
                        else float(bias),
                    },
                    "plugins",
                )
            except Exception:
                pass
            return out

        x_list = x if isinstance(x, list) else [float(x)]
        inten_f = float(inten.detach().to("cpu").item()) if hasattr(inten, "detach") else float(inten)
        bias_f = float(bias.detach().to("cpu").item()) if hasattr(bias, "detach") else float(bias)
        out_vals = [math.copysign(math.log1p(abs(xv) * inten_f), xv) + bias_f for xv in x_list]
        try:
            report("neuron", "entropy_burst_forward", {"intensity": inten_f, "bias": bias_f}, "plugins")
        except Exception:
            pass
        return out_vals if len(out_vals) != 1 else out_vals[0]


__all__ = ["EntropyBurstNeuronPlugin"]
