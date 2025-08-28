from __future__ import annotations

from typing import Any, Dict

from ..reporter import report


class QualityAwareTrainPlugin:
    """Adjust learning rate each walk based on current target quality."""

    def on_init(self, brain: "Brain", wanderer: "Wanderer", config: Dict[str, Any]) -> None:  # noqa: D401
        pass

    def before_walk(self, brain: "Brain", wanderer: "Wanderer", i: int) -> Dict[str, Any]:
        try:
            tgt = float(wanderer._target_provider(None))  # type: ignore[call-arg]
        except Exception:
            tgt = 0.0
        lr = 1e-3 + max(0.0, min(1.0, tgt)) * 5e-4
        try:
            report("training", "qualityaware_before", {"walk": i, "lr": lr, "target": tgt}, "brain")
        except Exception:
            pass
        return {"lr": float(lr)}


__all__ = ["QualityAwareTrainPlugin"]

