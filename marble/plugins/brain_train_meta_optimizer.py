"""MetaOptimizerTrainPlugin adjusts learning rate via a learnable scale.

The plugin exposes a single ``meta_lr`` parameter through
``expose_learnable_params``. Before every walk, the provided learning rate is
multiplied by this meta parameter, allowing the training loop itself to learn
how aggressively it should update weights.
"""

from __future__ import annotations

from typing import Any, Dict

from ..wanderer import expose_learnable_params
from ..reporter import report


@expose_learnable_params
def _meta_params(wanderer, meta_lr: float = 1.0):
    """Return a learnable scaling for the base learning rate."""
    return meta_lr


class MetaOptimizerTrainPlugin:
    """Brain-train plugin that learns a global LR multiplier."""

    def on_init(self, brain: "Brain", wanderer: "Wanderer", config: Dict[str, Any]) -> None:  # noqa: D401
        report("training", "meta_optimizer_init", {}, "brain")

    def before_walk(self, brain: "Brain", wanderer: "Wanderer", i: int) -> Dict[str, Any]:
        meta_lr_t = _meta_params(wanderer)
        try:
            scale = float(meta_lr_t.detach().to("cpu").item())
        except Exception:
            scale = 1.0
        base = getattr(wanderer, "_neuro_cfg", {}).get("lr", 1e-3)
        try:
            base = float(base)
        except Exception:
            base = 1e-3
        lr = base * scale
        report("training", "meta_optimizer_before", {"walk": i, "lr": lr}, "brain")
        return {"lr": float(lr)}


__all__ = ["MetaOptimizerTrainPlugin"]

