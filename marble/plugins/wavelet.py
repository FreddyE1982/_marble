from __future__ import annotations

"""Wavelet neuron plugin using a Morlet-like transform."""

from typing import Any, Tuple

from ..wanderer import expose_learnable_params


class WaveletNeuronPlugin:
    """Apply a Gaussian-windowed cosine (Morlet) wavelet."""

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer: "Wanderer",
        *,
        wav_scale: float = 1.0,
        wav_shift: float = 0.0,
        wav_sigma: float = 1.0,
        wav_freq: float = 1.0,
        wav_bias: float = 0.0,
    ) -> Tuple[Any, Any, Any, Any, Any]:
        return wav_scale, wav_shift, wav_sigma, wav_freq, wav_bias

    def forward(self, neuron: "Neuron", input_value=None):
        x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)

        scale, shift, sigma, freq, bias = 1.0, 0.0, 1.0, 1.0, 0.0
        wanderer = neuron._plugin_state.get("wanderer")
        if wanderer is not None:
            try:
                scale, shift, sigma, freq, bias = self._params(wanderer)
            except Exception:
                pass

        torch = getattr(neuron, "_torch", None)
        if torch is not None and neuron._is_torch_tensor(x):
            xc = x - shift
            y = scale * torch.exp(-0.5 * (xc / sigma) ** 2) * torch.cos(freq * xc) + bias
            return y

        import math

        x_list = x if isinstance(x, list) else [float(x)]
        scale_f = float(scale if not hasattr(scale, "detach") else scale.detach().to("cpu").item())
        shift_f = float(shift if not hasattr(shift, "detach") else shift.detach().to("cpu").item())
        sigma_f = float(sigma if not hasattr(sigma, "detach") else sigma.detach().to("cpu").item())
        freq_f = float(freq if not hasattr(freq, "detach") else freq.detach().to("cpu").item())
        bias_f = float(bias if not hasattr(bias, "detach") else bias.detach().to("cpu").item())
        out = [
            scale_f * math.exp(-0.5 * ((v - shift_f) / sigma_f) ** 2) * math.cos(freq_f * (v - shift_f)) + bias_f
            for v in map(float, x_list)
        ]
        return out if len(out) != 1 else out[0]


__all__ = ["WaveletNeuronPlugin"]

