from __future__ import annotations
from typing import Any, Dict
from ..wanderer import expose_learnable_params
from ..reporter import report

@expose_learnable_params
def _shift_params(wanderer, shift_lr: float = 0.1):
    return shift_lr,

class BiasShiftPlugin:
    """Shift biases of visited neurons according to final loss sign."""

    def on_walk_end(self, wanderer: "Wanderer", stats: Dict[str, Any]) -> None:
        shift_t, = _shift_params(wanderer)
        try:
            shift = float(shift_t.detach().to("cpu").item())
        except Exception:
            shift = 0.1
        try:
            loss = float(stats.get("loss", 0.0))
            sign = 1.0 if loss < 0 else -1.0
        except Exception:
            sign = 1.0
        for n in getattr(wanderer, "_visited", []):
            try:
                n.bias = float(getattr(n, "bias", 0.0)) + shift * sign
            except Exception:
                pass
        try:
            report("neuroplasticity", "bias_shift", {"shift": shift, "sign": float(sign)}, "plugins")
        except Exception:
            pass
        return None

__all__ = ["BiasShiftPlugin"]
