"""AgePruneRoutine cools temperature when old neurons dominate."""

from __future__ import annotations

from typing import Any, Dict
import torch

from ..wanderer import expose_learnable_params
from ..reporter import report
from .selfattention_metric_utils import metric_factor


@expose_learnable_params
def _age_prune_params(wanderer, max_age: float = 100.0, cool_temp: float = 0.2):
    return max_age, cool_temp


class AgePruneRoutine:
    """Lower temperature if any neuron age exceeds ``max_age``."""

    def after_step(
        self,
        selfattention: "SelfAttention",
        reporter_ro: Any,
        wanderer: "Wanderer",
        step_index: int,
        ctx: Dict[str, Any],
    ):
        max_t, cool_t = _age_prune_params(wanderer)
        try:
            max_age = float(max_t.detach().to("cpu").item())
        except Exception:
            max_age = 100.0
        try:
            cool_temp = float(cool_t.detach().to("cpu").item())
        except Exception:
            cool_temp = 0.2
        factor = metric_factor(ctx, "age_prune")
        max_age *= 1.0 + factor
        cool_temp *= factor
        for n in list(getattr(wanderer.brain, "neurons", {}).values()):
            info = selfattention.get_neuron_report(n)
            try:
                age = float(info.get("age", 0.0))
            except Exception:
                continue
            if age > max_age:
                try:
                    selfattention.set_param("temperature", cool_temp)
                except Exception:
                    pass
                report("selfattention", "age_prune", {"step": step_index, "age": age}, "events")
                break
        return None


__all__ = ["AgePruneRoutine"]
