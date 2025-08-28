from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..marblemain import register_brain_train_type
from ..reporter import report


class WarmupDecayTrainPlugin:
    """Per-walk scheduler that warms up LR then decays it, and grows steps.

    Config (in `Brain.train(..., type_name='warmup_decay', ...)` or stacked):
      - warmup_walks (int, default 3)
      - base_lr (float, default 1e-2)
      - peak_lr (float, default 5e-2)
      - decay (float, default 0.9) multiplicative each walk after warmup
      - start_steps (int, default 2) and step_increment (int, default 1)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(config or {})
        self.warmup = int(cfg.get("warmup_walks", 3))
        self.base_lr = float(cfg.get("base_lr", 1e-2))
        self.peak_lr = float(cfg.get("peak_lr", 5e-2))
        self.decay = float(cfg.get("decay", 0.9))
        self.start_steps = int(cfg.get("start_steps", 2))
        self.step_inc = int(cfg.get("step_increment", 1))

    def on_init(self, brain: "Brain", wanderer: "Wanderer", config: Dict[str, Any]) -> None:
        try:
            report("training", "warmup_decay_init", {"warmup": self.warmup}, "brain")
        except Exception:
            pass

    def choose_start(self, brain: "Brain", wanderer: "Wanderer", i: int):
        return None

    def before_walk(self, brain: "Brain", wanderer: "Wanderer", i: int) -> Dict[str, Any]:
        steps = self.start_steps + i * self.step_inc
        if i < self.warmup:
            # Linear warmup
            t = (i + 1) / float(max(1, self.warmup))
            lr = self.base_lr + t * (self.peak_lr - self.base_lr)
        else:
            # Exponential decay from peak
            k = i - self.warmup
            lr = self.peak_lr * (self.decay ** k)
        try:
            report("training", "warmup_decay_before", {"walk": i, "steps": steps, "lr": lr}, "brain")
        except Exception:
            pass
        return {"max_steps": int(max(1, steps)), "lr": float(lr)}

    def after_walk(self, brain: "Brain", wanderer: "Wanderer", i: int, stats: Dict[str, Any]) -> None:
        try:
            report("training", "warmup_decay_after", {"walk": i, "loss": stats.get("loss")}, "brain")
        except Exception:
            pass

    def on_end(self, brain: "Brain", wanderer: "Wanderer", history: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"warmup_decay": {"walks": len(history), "final_loss": history[-1].get("loss") if history else None}}


try:
    register_brain_train_type("warmup_decay", WarmupDecayTrainPlugin())
except Exception:
    pass

__all__ = ["WarmupDecayTrainPlugin"]
