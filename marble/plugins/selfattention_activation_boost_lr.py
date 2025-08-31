"""ActivationBoostLRRoutine increases learning rate when activations are low."""

from __future__ import annotations

from typing import Any, Dict, List
import torch

from ..wanderer import expose_learnable_params
from ..reporter import report
from .selfattention_metric_utils import metric_factor


@expose_learnable_params
def _act_boost_params(wanderer, threshold: float = 0.1, boost: float = 2.0):
    return threshold, boost


class ActivationBoostLRRoutine:
    """Boost ``lr_override`` if avg activation < ``threshold``."""

    def after_step(
        self,
        selfattention: "SelfAttention",
        reporter_ro: Any,
        wanderer: "Wanderer",
        step_index: int,
        ctx: Dict[str, Any],
    ):
        thr_t, boost_t = _act_boost_params(wanderer)
        try:
            thr = float(thr_t.detach().to("cpu").item())
        except Exception:
            thr = 0.1
        try:
            boost = float(boost_t.detach().to("cpu").item())
        except Exception:
            boost = 2.0
        factor = metric_factor(ctx, "activation_boost_lr")
        thr *= 1.0 + factor
        boost *= factor
        acts: List[float] = []
        for n in list(getattr(wanderer.brain, "neurons", {}).values()):
            info = selfattention.get_neuron_report(n)
            val = info.get("activation")
            if val is not None:
                try:
                    acts.append(abs(float(val)))
                except Exception:
                    pass
        if acts and sum(acts) / len(acts) < thr:
            cur = float(selfattention.get_param("lr_override", 0.0) or 0.0)
            selfattention.set_param("lr_override", cur * boost if cur else boost)
            report("selfattention", "activation_boost_lr", {"step": step_index, "avg": sum(acts) / len(acts)}, "events")
        return None


__all__ = ["ActivationBoostLRRoutine"]
