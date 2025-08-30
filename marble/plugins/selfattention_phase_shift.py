"""PhaseShiftRoutine tweaks attention temperature via a learnable phase shift."""

from __future__ import annotations

from typing import Any, Dict

from ..wanderer import expose_learnable_params
from ..reporter import report


@expose_learnable_params
def _phase_param(wanderer, phase_shift: float = 0.0):
    return phase_shift


class PhaseShiftRoutine:
    """Adjust temperature using a learnable phase shift."""

    def after_step(self, selfattention: "SelfAttention", reporter_ro: Any, wanderer: "Wanderer", step_index: int, ctx: Dict[str, Any]):
        ph_t = _phase_param(wanderer)
        try:
            ph = float(ph_t.detach().to("cpu").item())
        except Exception:
            ph = 0.0
        try:
            selfattention.set_param("temperature", 1.0 + ph)
        except Exception:
            pass
        report("selfattention", "phase_shift", {"step": step_index, "phase": ph}, "events")
        return None


__all__ = ["PhaseShiftRoutine"]

