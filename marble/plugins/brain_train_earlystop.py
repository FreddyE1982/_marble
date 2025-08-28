from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..reporter import report


class EarlyStoppingTrainPlugin:
    """Stop Brain training early when loss stalls.

    Config (passed at construction time):
      - patience (int, default 3): number of consecutive non-improving walks to tolerate.
      - min_delta (float, default 0.0): minimum improvement to reset patience.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(config or {})
        self.patience = int(cfg.get("patience", 3))
        self.min_delta = float(cfg.get("min_delta", 0.0))
        self.best: Optional[float] = None
        self.count = 0
        self.stopped = False

    def on_init(self, brain: "Brain", wanderer: "Wanderer", config: Dict[str, Any]) -> None:
        try:
            report("training", "earlystop_init", {"patience": self.patience}, "brain")
        except Exception:
            pass

    def after_walk(self, brain: "Brain", wanderer: "Wanderer", i: int, stats: Dict[str, Any]) -> Dict[str, Any]:
        loss = float(stats.get("loss", 0.0))
        if self.best is None or loss < self.best - self.min_delta:
            self.best = loss
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.stopped = True
                try:
                    report("training", "earlystop_trigger", {"walk": i, "loss": loss}, "brain")
                except Exception:
                    pass
                return {"stop": True}
        try:
            report(
                "training",
                "earlystop_after",
                {"walk": i, "loss": loss, "best": self.best, "count": self.count},
                "brain",
            )
        except Exception:
            pass
        return {}

    def on_end(self, brain: "Brain", wanderer: "Wanderer", history: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"early_stopped": self.stopped, "best_loss": self.best}


__all__ = ["EarlyStoppingTrainPlugin"]

