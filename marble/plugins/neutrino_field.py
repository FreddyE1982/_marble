from __future__ import annotations

"""Neutrino field neuron plugin.

Applies a hypothetical neutrino field interaction that dampens values based on
magnitude while injecting directionality to encourage emergent pathways."""

from typing import Any, Tuple
import math

from ..wanderer import expose_learnable_params
from ..reporter import report


class NeutrinoFieldNeuronPlugin:
    """Interact activations with an imaginary neutrino field."""

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer: "Wanderer",
        *,
        field_strength: float = 1.0,
        field_decay: float = 0.1,
    ) -> Tuple[Any, Any]:
        return (field_strength, field_decay)

    def forward(self, neuron: "Neuron", input_value=None):
        x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)

        strength, decay = 1.0, 0.1
        wanderer = neuron._plugin_state.get("wanderer")
        if wanderer is not None:
            try:
                strength, decay = self._params(wanderer)
            except Exception:
                pass

        torch = getattr(neuron, "_torch", None)
        if torch is not None and neuron._is_torch_tensor(x):
            sign = torch.sign(x)
            out = sign * strength * torch.exp(-torch.abs(x) * decay)
            try:
                report(
                    "neuron",
                    "neutrino_field_forward",
                    {
                        "strength": float(strength.detach().to("cpu").item())
                        if hasattr(strength, "detach")
                        else float(strength),
                        "decay": float(decay.detach().to("cpu").item())
                        if hasattr(decay, "detach")
                        else float(decay),
                    },
                    "plugins",
                )
            except Exception:
                pass
            return out

        x_list = x if isinstance(x, list) else [float(x)]
        strength_f = float(strength.detach().to("cpu").item()) if hasattr(strength, "detach") else float(strength)
        decay_f = float(decay.detach().to("cpu").item()) if hasattr(decay, "detach") else float(decay)
        out_vals = [math.copysign(strength_f * math.exp(-abs(xv) * decay_f), xv) for xv in x_list]
        try:
            report("neuron", "neutrino_field_forward", {"strength": strength_f, "decay": decay_f}, "plugins")
        except Exception:
            pass
        return out_vals if len(out_vals) != 1 else out_vals[0]


__all__ = ["NeutrinoFieldNeuronPlugin"]
