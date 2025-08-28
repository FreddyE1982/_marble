from __future__ import annotations

from typing import Any, Dict

from ..marblemain import register_brain_train_type


class CurriculumTrainPlugin:
    """Brain-train plugin that increases walk max_steps across walks.

    Config keys (read from the `config` dict passed to on_init):
    - start_steps (default 1)
    - step_increment (default 1)
    - max_cap (optional int): if provided, cap max_steps to this value.
    """
    def on_init(self, brain: "Brain", wanderer: "Wanderer", config: Dict[str, Any]) -> None:
        cfg = dict(config or {})
        self.start_steps = int(cfg.get("start_steps", 1))
        self.inc = int(cfg.get("step_increment", 1))
        self.cap = cfg.get("max_cap")

    def before_walk(self, brain: "Brain", wanderer: "Wanderer", i: int) -> Dict[str, Any]:
        ms = self.start_steps + i * self.inc
        if self.cap is not None:
            try:
                ms = min(ms, int(self.cap))
            except Exception:
                pass
        return {"max_steps": int(ms)}


try:
    register_brain_train_type("curriculum", CurriculumTrainPlugin())
except Exception:
    pass

__all__ = ["CurriculumTrainPlugin"]
