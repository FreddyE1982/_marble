from __future__ import annotations
from typing import Any, Dict
from ..wanderer import expose_learnable_params
from ..reporter import report

@expose_learnable_params
def _entropy_params(wanderer, entropy_threshold: float = 0.5, high_temp: float = 2.0, low_temp: float = 0.5):
    return entropy_threshold, high_temp, low_temp

class EntropyRoutingRoutine:
    """Adjust wanderer temperature based on loss change entropy."""

    def on_init(self, selfattention: "SelfAttention") -> None:  # pragma: no cover - trivial
        pass

    def after_step(self, selfattention: "SelfAttention", reporter_ro: Any, wanderer: "Wanderer", step_index: int, ctx: Dict[str, Any]):
        thr_t, high_t, low_t = _entropy_params(wanderer)
        try:
            thr = float(thr_t.detach().to("cpu").item())
            hi = float(high_t.detach().to("cpu").item())
            lo = float(low_t.detach().to("cpu").item())
        except Exception:
            thr, hi, lo = 0.5, 2.0, 0.5
        try:
            cur = float(ctx.get("cur_loss_tensor").detach().to("cpu").item()) if ctx.get("cur_loss_tensor") is not None else None
        except Exception:
            cur = None
        prev = None
        try:
            last = reporter_ro.item(f"step_{max(1, getattr(wanderer, '_global_step_counter', 1)) - 1}", "wanderer_steps", "logs")
            if isinstance(last, dict) and "current_loss" in last:
                prev = float(last.get("current_loss"))
        except Exception:
            prev = None
        if cur is None or prev is None:
            return None
        entropy = abs(cur - prev)
        try:
            report("selfattention", "entropy", {"step": int(step_index), "entropy": float(entropy)}, "events")
        except Exception:
            pass
        try:
            if entropy > thr:
                selfattention.set_param("temperature", hi)
            else:
                selfattention.set_param("temperature", lo)
        except Exception:
            pass
        return None

__all__ = ["EntropyRoutingRoutine"]
