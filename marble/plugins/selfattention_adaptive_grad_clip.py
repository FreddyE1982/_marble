from __future__ import annotations

from typing import Any, Dict

from ..reporter import report


class AdaptiveGradClipRoutine:
    """Adjusts gradient clipping based on step loss spikes.

    If current per-step loss grows by more than `threshold_ratio` over the
    previous step, set gradient clipping (method='norm', max_norm).
    Fields (constructor/defaults): threshold_ratio=1.5, max_norm=1.0, cooldown=5
    """

    def __init__(self, threshold_ratio: float = 1.5, max_norm: float = 1.0, cooldown: int = 5) -> None:
        self.threshold_ratio = float(threshold_ratio)
        self.max_norm = float(max_norm)
        self.cooldown = int(max(0, cooldown))
        self._since = 0

    def on_init(self, selfattention: "SelfAttention") -> None:
        self._since = 0

    def after_step(self, selfattention: "SelfAttention", reporter_ro: Any, wanderer: "Wanderer", step_index: int, ctx: Dict[str, Any]):
        try:
            cur = float(ctx.get("cur_loss_tensor").detach().to("cpu").item()) if ctx.get("cur_loss_tensor") is not None else None
        except Exception:
            cur = None
        prev = None
        try:
            last = reporter_ro.item(f"step_{max(1, getattr(wanderer, '_global_step_counter', 1)) - 1}", "wanderer_steps", "logs")
            if isinstance(last, dict) and ("current_loss" in last):
                prev = float(last.get("current_loss"))
        except Exception:
            prev = None
        if cur is None or prev is None:
            return None
        if prev <= 0.0:
            return None
        ratio = cur / prev
        if ratio >= self.threshold_ratio and self._since <= 0:
            try:
                selfattention.set_param("_grad_clip", {"method": "norm", "max_norm": float(self.max_norm), "norm_type": 2.0})
                report("selfattention", "gradclip_enable", {"ratio": ratio, "max_norm": float(self.max_norm)}, "events")
            except Exception:
                pass
            self._since = self.cooldown
        else:
            self._since = max(0, self._since - 1)
        return None


__all__ = ["AdaptiveGradClipRoutine"]
