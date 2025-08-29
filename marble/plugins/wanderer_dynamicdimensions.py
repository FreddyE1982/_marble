from __future__ import annotations

from typing import Any, Dict

from ..reporter import report


class DynamicDimensionsPlugin:
    """Adds a temporary dimension and evaluates its benefit."""

    def __init__(self) -> None:
        self._pending = False
        self._baseline: float | None = None
        self._walks = 0

    def on_walk_end(self, wanderer: "Wanderer", stats: Dict[str, Any]) -> None:
        self._walks += 1
        cfg = getattr(wanderer, "_neuro_cfg", {}) or {}
        every = int(cfg.get("dynamicdims_every", 10))
        min_neurons = int(cfg.get("dynamicdims_min_neurons", 5))
        thresh = float(cfg.get("dynamicdims_improve_thresh", 0.0))
        brain = wanderer.brain
        if self._pending:
            count = 0
            try:
                for pos in brain.neurons.keys():
                    if len(pos) == brain.n and pos[-1] != (0 if brain.mode == "grid" else 0.0):
                        count += 1
            except Exception:
                count = 0
            if count >= min_neurons:
                loss = float(stats.get("loss", 0.0))
                if self._baseline is not None and loss > self._baseline - thresh:
                    try:
                        brain.remove_last_dimension()
                        report("wanderer", "dynadim_remove", {"loss": loss}, "plugins")
                    except Exception:
                        pass
                self._pending = False
                self._baseline = None
            return
        if self._walks % every == 0:
            self._baseline = float(stats.get("loss", 0.0))
            try:
                brain.add_dimension()
                report("wanderer", "dynadim_add", {"walk": self._walks}, "plugins")
            except Exception:
                pass
            self._pending = True

__all__ = ["DynamicDimensionsPlugin"]
