from __future__ import annotations

"""Temporal fission neuron plugin.

Splits the incoming signal into divergent temporal shards whose interplay is
meant to provoke emergent temporal patterns."""

from typing import Any, Tuple
import math

from ..wanderer import expose_learnable_params
from ..reporter import report


class TemporalFissionNeuronPlugin:
    """Split activations across fictive time shards."""

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer: "Wanderer",
        *,
        fiss_amp: float = 1.0,
        fiss_phase: float = 0.0,
    ) -> Tuple[Any, Any]:
        return (fiss_amp, fiss_phase)

    def forward(self, neuron: "Neuron", input_value=None):
        x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)

        amp, phase = 1.0, 0.0
        wanderer = neuron._plugin_state.get("wanderer")
        if wanderer is not None:
            try:
                amp, phase = self._params(wanderer)
            except Exception:
                pass

        torch = getattr(neuron, "_torch", None)
        if torch is not None and neuron._is_torch_tensor(x):
            y1 = torch.sin(x + phase) * amp
            y2 = torch.cos(x - phase) * (2 - amp)
            out = y1 + y2
            try:
                report(
                    "neuron",
                    "temporal_fission_forward",
                    {
                        "amp": float(amp.detach().to("cpu").item()) if hasattr(amp, "detach") else float(amp),
                        "phase": float(phase.detach().to("cpu").item())
                        if hasattr(phase, "detach")
                        else float(phase),
                    },
                    "plugins",
                )
            except Exception:
                pass
            return out

        x_list = x if isinstance(x, list) else [float(x)]
        amp_f = float(amp.detach().to("cpu").item()) if hasattr(amp, "detach") else float(amp)
        phase_f = float(phase.detach().to("cpu").item()) if hasattr(phase, "detach") else float(phase)
        out_vals = [math.sin(xv + phase_f) * amp_f + math.cos(xv - phase_f) * (2 - amp_f) for xv in x_list]
        try:
            report("neuron", "temporal_fission_forward", {"amp": amp_f, "phase": phase_f}, "plugins")
        except Exception:
            pass
        return out_vals if len(out_vals) != 1 else out_vals[0]


__all__ = ["TemporalFissionNeuronPlugin"]
