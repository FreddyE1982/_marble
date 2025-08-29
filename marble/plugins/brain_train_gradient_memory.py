"""GradientMemoryTrainPlugin adapts learning rate based on loss history.

The plugin exposes a ``memory_decay`` parameter controlling how quickly the
historical loss influence fades. After each walk, the plugin updates an internal
moving average of the losses and lowers the learning rate if the average grows.
"""

from __future__ import annotations

from typing import Any, Dict

from ..wanderer import expose_learnable_params
from ..reporter import report


@expose_learnable_params
def _memory_param(wanderer, memory_decay: float = 0.9):
    return memory_decay


class GradientMemoryTrainPlugin:
    """Brain-train plugin tracking an exponential loss memory."""

    def __init__(self) -> None:
        self.avg: float = 0.0

    def on_init(self, brain: "Brain", wanderer: "Wanderer", config: Dict[str, Any]) -> None:  # noqa: D401
        self.avg = 0.0

    def after_walk(self, brain: "Brain", wanderer: "Wanderer", i: int, stats: Dict[str, Any]) -> None:
        decay_t = _memory_param(wanderer)
        try:
            decay = float(decay_t.detach().to("cpu").item())
        except Exception:
            decay = 0.9
        loss = float(stats.get("loss", 0.0))
        self.avg = decay * self.avg + (1 - decay) * loss
        report("training", "gradmem_after", {"walk": i, "avg": self.avg}, "brain")
        if self.avg > loss:
            # reduce LR slightly if losses trend upward
            lr = max(1e-6, getattr(wanderer, "_neuro_cfg", {}).get("lr", 1e-3) * 0.9)
            wanderer._neuro_cfg["lr"] = lr


__all__ = ["GradientMemoryTrainPlugin"]

