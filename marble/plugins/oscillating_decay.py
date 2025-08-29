from __future__ import annotations

"""Oscillating decay neuron plugin.

Generates damped oscillations by combining sinusoidal responses with an
exponential decay envelope. Both frequency and decay rate are learnable.
"""

from typing import Any, Tuple
import math

from ..wanderer import expose_learnable_params
from ..reporter import report


class OscillatingDecayNeuronPlugin:
    """Damped sinusoidal activation with learnable parameters."""

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer: "Wanderer", *, osc_freq: float = 1.0, osc_decay: float = 0.1
    ) -> Tuple[Any, Any]:
        return (osc_freq, osc_decay)

    def forward(self, neuron: "Neuron", input_value=None):
        x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)

        freq = 1.0
        decay = 0.1
        wanderer = neuron._plugin_state.get("wanderer")
        if wanderer is not None:
            try:
                freq, decay = self._params(wanderer)
            except Exception:
                pass

        torch = getattr(neuron, "_torch", None)
        if torch is not None and neuron._is_torch_tensor(x):
            y = x * torch.exp(-decay * torch.abs(x)) * torch.sin(freq * x)
            try:
                report(
                    "neuron",
                    "oscillating_decay_forward",
                    {
                        "freq": float(freq.detach().to("cpu").item()) if hasattr(freq, "detach") else float(freq),
                        "decay": float(decay.detach().to("cpu").item()) if hasattr(decay, "detach") else float(decay),
                    },
                    "plugins",
                )
            except Exception:
                pass
            return y

        x_list = x if isinstance(x, list) else [float(x)]
        freq_f = float(freq.detach().to("cpu").item()) if hasattr(freq, "detach") else float(freq)
        decay_f = float(decay.detach().to("cpu").item()) if hasattr(decay, "detach") else float(decay)
        out = [v * math.exp(-decay_f * abs(v)) * math.sin(freq_f * v) for v in x_list]
        try:
            report(
                "neuron",
                "oscillating_decay_forward",
                {"freq": freq_f, "decay": decay_f},
                "plugins",
            )
        except Exception:
            pass
        return out if len(out) != 1 else out[0]


__all__ = ["OscillatingDecayNeuronPlugin"]

