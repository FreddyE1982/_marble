from __future__ import annotations

from typing import Any, Dict

from ..reporter import report
from ..wanderer import expose_learnable_params


@expose_learnable_params
def _agc_params(
    wanderer,
    threshold_ratio: float = 1.5,
    max_norm: float = 1.0,
    cooldown: float = 5.0,
):
    return threshold_ratio, max_norm, cooldown


class AdaptiveGradClipRoutine:
    """Adjust gradient clipping based on loss spikes.

    Parameters are exposed as learnables via ``expose_learnable_params`` to
    ensure compatibility with ``AutoPlugin`` and future tuning.
    """

    def __init__(self) -> None:
        self._since = 0

    def on_init(self, selfattention: "SelfAttention") -> None:
        self._since = 0

    def after_step(self, selfattention: "SelfAttention", reporter_ro: Any, wanderer: "Wanderer", step_index: int, ctx: Dict[str, Any]):
        thr_t, max_t, cd_t = _agc_params(wanderer)
        try:
            thr = float(thr_t.detach().to("cpu").item())
            max_norm = float(max_t.detach().to("cpu").item())
            cooldown = int(float(cd_t.detach().to("cpu").item()))
        except Exception:
            thr, max_norm, cooldown = 1.5, 1.0, 5
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
        if ratio >= thr and self._since <= 0:
            try:
                selfattention.set_param(
                    "_grad_clip",
                    {"method": "norm", "max_norm": float(max_norm), "norm_type": 2.0},
                )
                report(
                    "selfattention",
                    "gradclip_enable",
                    {"ratio": ratio, "max_norm": float(max_norm)},
                    "events",
                )
            except Exception:
                pass
            self._since = cooldown
        else:
            self._since = max(0, self._since - 1)
        return None


__all__ = ["AdaptiveGradClipRoutine"]
