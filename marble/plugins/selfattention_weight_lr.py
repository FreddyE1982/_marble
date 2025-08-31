from __future__ import annotations

from typing import Any, Dict, List

from ..wanderer import expose_learnable_params
from ..reporter import report
from .selfattention_metric_utils import metric_factor


@expose_learnable_params
def _weight_lr_params(
    wanderer,
    num_neurons: float = 3.0,
    start_index: float = 0.0,
    threshold: float = 1.0,
    scale: float = 0.5,
):
    return num_neurons, start_index, threshold, scale


class WeightLRRoutine:
    """Adjust learning rate based on average absolute neuron weight."""

    def after_step(
        self,
        selfattention: "SelfAttention",
        reporter_ro: Any,
        wanderer: "Wanderer",
        step_index: int,
        ctx: Dict[str, Any],
    ):
        n_t, s_t, thr_t, sc_t = _weight_lr_params(wanderer)
        try:
            count = max(1, int(n_t.detach().to("cpu").item()))
            start = int(s_t.detach().to("cpu").item())
            thr = float(thr_t.detach().to("cpu").item())
            scale = float(sc_t.detach().to("cpu").item())
        except Exception:
            count, start, thr, scale = 3, 0, 1.0, 0.5
        mf = metric_factor(ctx, "weight_lr")
        thr *= 1.0 + mf
        scale *= mf
        neurons = list(getattr(wanderer.brain, "neurons", {}).values())
        if not neurons:
            return None
        neurons.sort(key=lambda n: (getattr(n, "position", ()), id(n)))
        start = start % len(neurons)
        chosen: List[Any] = [neurons[(start + i) % len(neurons)] for i in range(count)]
        weights: List[float] = []
        for n in chosen:
            info = selfattention.get_neuron_report(n)
            try:
                weights.append(abs(float(info.get("weight", 0.0))))
            except Exception:
                pass
        if not weights:
            return None
        avg = sum(weights) / len(weights)
        try:
            report("selfattention", "weight_lr", {"step": step_index, "avg": avg}, "events")
        except Exception:
            pass
        cur_lr = float(getattr(wanderer, "current_lr", 0.01) or 0.01)
        if avg > thr:
            selfattention.set_param("lr_override", cur_lr * scale)
        return None


__all__ = ["WeightLRRoutine"]

