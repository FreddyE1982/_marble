"""TemporalEchoRoutine injects decaying echo into attention temperature."""

from __future__ import annotations

from typing import Any, Dict

from ..wanderer import expose_learnable_params
from ..reporter import report
from .selfattention_metric_utils import metric_factor


@expose_learnable_params
def _echo_param(wanderer, echo_decay: float = 0.5):
    return echo_decay


class TemporalEchoRoutine:
    """Apply a decaying echo to temperature each step."""

    def after_step(self, selfattention: "SelfAttention", reporter_ro: Any, wanderer: "Wanderer", step_index: int, ctx: Dict[str, Any]):
        ed_t = _echo_param(wanderer)
        try:
            ed = float(ed_t.detach().to("cpu").item())
        except Exception:
            ed = 0.5
        ed *= metric_factor(ctx, "temporal_echo")
        try:
            base = float(selfattention.get_param("temperature", 1.0))
            selfattention.set_param("temperature", base * (1.0 - abs(ed)))
        except Exception:
            pass
        report("selfattention", "temporal_echo", {"step": step_index, "decay": ed}, "events")
        return None


__all__ = ["TemporalEchoRoutine"]

