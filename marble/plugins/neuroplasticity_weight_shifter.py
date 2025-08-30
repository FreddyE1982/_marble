"""WeightShifterPlugin adds a learnable shift to traversed synapse weights."""

from __future__ import annotations

from typing import Any

from ..wanderer import expose_learnable_params
from ..reporter import report


@expose_learnable_params
def _shift_param(wanderer, shift_amount: float = 0.0):
    return shift_amount


class WeightShifterPlugin:
    """Shift synapse weights by a learnable amount on each step."""

    def on_step(self, wanderer: "Wanderer", current: "Neuron", syn: "Synapse", direction: str, step_index: int, out_value: Any) -> None:
        sh_t = _shift_param(wanderer)
        try:
            sh = float(sh_t.detach().to("cpu").item())
        except Exception:
            sh = 0.0
        try:
            syn.weight = float(getattr(syn, "weight", 0.0)) + sh
            report("neuroplasticity", "weight_shift", {"step": int(step_index), "shift": sh}, "plugins")
        except Exception:
            pass
        return None


__all__ = ["WeightShifterPlugin"]

