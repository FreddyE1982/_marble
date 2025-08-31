from __future__ import annotations

from typing import Any, Dict

from ..wanderer import expose_learnable_params
from ..reporter import report
from .selfattention_metric_utils import metric_factor


@expose_learnable_params
def _noise_params(
    wanderer,
    noise_variance: float = 0.05,
    spatial_factor: float = 0.5,
):
    """Return learnable tensors for noise variance and spatial factor."""
    return noise_variance, spatial_factor


class ContextAwareNoiseRoutine:
    """Model sensor noise with learnable parameters and adapt LR accordingly."""

    def on_init(self, selfattention: "SelfAttention") -> None:  # pragma: no cover - simple state init
        pass

    def after_step(
        self,
        selfattention: "SelfAttention",
        reporter_ro: Any,
        wanderer: "Wanderer",
        step_index: int,
        ctx: Dict[str, Any],
    ):
        var_t, spatial_t = _noise_params(wanderer)
        try:
            noise_score = float((var_t * spatial_t).detach().to("cpu").item())
        except Exception:
            return None
        mf = metric_factor(ctx, "noise_profiler")
        noise_score *= 1.0 + mf
        try:
            report(
                "selfattention",
                "context_noise",
                {"step": int(step_index), "score": float(noise_score)},
                "events",
            )
        except Exception:
            pass
        base_lr = selfattention.get_param("lr_override") or selfattention.get_param("current_lr") or 1e-3
        try:
            base_lr = float(base_lr)
        except Exception:
            base_lr = 1e-3
        if noise_score > 0.05:
            new_lr = max(1e-5, base_lr * 0.9)
        else:
            new_lr = min(5e-3, base_lr * 1.05)
        new_lr *= mf
        return {"lr_override": float(new_lr)}


__all__ = ["ContextAwareNoiseRoutine"]

