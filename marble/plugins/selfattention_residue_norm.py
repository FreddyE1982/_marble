"""ResidueNormRoutine nudges temperature toward a bias with learnable residue."""

from __future__ import annotations

from typing import Any, Dict

from ..wanderer import expose_learnable_params
from ..reporter import report
from .selfattention_metric_utils import metric_factor


@expose_learnable_params
def _residue_param(wanderer, residue_bias: float = 0.0):
    return residue_bias


class ResidueNormRoutine:
    """Slowly push temperature to a learnable residue bias."""

    def after_step(self, selfattention: "SelfAttention", reporter_ro: Any, wanderer: "Wanderer", step_index: int, ctx: Dict[str, Any]):
        rb_t = _residue_param(wanderer)
        try:
            rb = float(rb_t.detach().to("cpu").item())
        except Exception:
            rb = 0.0
        rb *= metric_factor(ctx, "residue_norm")
        try:
            cur = float(selfattention.get_param("temperature", 1.0))
            selfattention.set_param("temperature", cur + rb)
        except Exception:
            pass
        report("selfattention", "residue_norm", {"step": step_index, "bias": rb}, "events")
        return None


__all__ = ["ResidueNormRoutine"]

