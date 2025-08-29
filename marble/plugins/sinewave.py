from __future__ import annotations

"""Sine wave neuron plugin with learnable amplitude, frequency, phase and bias.

All parameters are exposed via :func:`expose_learnable_params` so the
``Wanderer`` tracks gradients for them.  The plugin computes

``y = amplitude * sin(frequency * x + phase) + bias``

for any input tensor ``x``.  Torch tensors are used when available, otherwise
pure Python fallbacks keep the behaviour functional during tests without
PyTorch.
"""

from typing import Any, Tuple
import math

from ..wanderer import expose_learnable_params
from ..reporter import report


class SineWaveNeuronPlugin:
    """Apply a learnable sine transformation to the neuron value."""

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer: "Wanderer",
        *,
        sine_amp: float = 1.0,
        sine_freq: float = 1.0,
        sine_phase: float = 0.0,
        sine_bias: float = 0.0,
    ) -> Tuple[Any, Any, Any, Any]:
        """Return learnable tensors for all plugin parameters."""

        return sine_amp, sine_freq, sine_phase, sine_bias

    def forward(self, neuron: "Neuron", input_value=None):
        x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)

        amp, freq, phase, bias = 1.0, 1.0, 0.0, 0.0
        wanderer = neuron._plugin_state.get("wanderer")
        if wanderer is not None:
            try:
                amp, freq, phase, bias = self._params(wanderer)
            except Exception:
                pass

        torch = getattr(neuron, "_torch", None)
        if torch is not None and neuron._is_torch_tensor(x):
            y = amp * torch.sin(freq * x + phase) + bias
            try:
                report(
                    "neuron",
                    "sine_forward",
                    {"amp": float(amp.detach().to("cpu").item()) if hasattr(amp, "detach") else float(amp)},
                    "plugins",
                )
            except Exception:
                pass
            return y

        x_list = x if isinstance(x, list) else [float(x)]
        amp_f = float(amp if not hasattr(amp, "detach") else amp.detach().to("cpu").item())
        freq_f = float(freq if not hasattr(freq, "detach") else freq.detach().to("cpu").item())
        phase_f = float(phase if not hasattr(phase, "detach") else phase.detach().to("cpu").item())
        bias_f = float(bias if not hasattr(bias, "detach") else bias.detach().to("cpu").item())
        out = [amp_f * math.sin(freq_f * float(v) + phase_f) + bias_f for v in x_list]
        try:
            report("neuron", "sine_forward", {"amp": amp_f}, "plugins")
        except Exception:
            pass
        return out if len(out) != 1 else out[0]


__all__ = ["SineWaveNeuronPlugin"]

