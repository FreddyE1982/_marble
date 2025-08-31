from __future__ import annotations

from typing import Any, Dict, List

from ..wanderer import expose_learnable_params
from ..reporter import report
from .selfattention_metric_utils import metric_factor


@expose_learnable_params
def _age_temp_params(
    wanderer,
    num_neurons: float = 3.0,
    start_index: float = 0.0,
    age_threshold: float = 1.0,
    young_temp: float = 2.0,
    old_temp: float = 0.5,
):
    return num_neurons, start_index, age_threshold, young_temp, old_temp


class AgeTemperatureRoutine:
    """Route temperature based on average neuron age."""

    def after_step(
        self,
        selfattention: "SelfAttention",
        reporter_ro: Any,
        wanderer: "Wanderer",
        step_index: int,
        ctx: Dict[str, Any],
    ):
        n_t, s_t, thr_t, y_t, o_t = _age_temp_params(wanderer)
        try:
            count = max(1, int(n_t.detach().to("cpu").item()))
            start = int(s_t.detach().to("cpu").item())
            thr = float(thr_t.detach().to("cpu").item())
            ytemp = float(y_t.detach().to("cpu").item())
            otemp = float(o_t.detach().to("cpu").item())
        except Exception:
            count, start, thr, ytemp, otemp = 3, 0, 1.0, 2.0, 0.5
        factor = metric_factor(ctx, "age_temperature")
        thr *= 1.0 + factor
        ytemp *= factor
        otemp *= factor
        neurons = list(getattr(wanderer.brain, "neurons", {}).values())
        if not neurons:
            return None
        neurons.sort(key=lambda n: (getattr(n, "position", ()), id(n)))
        start = start % len(neurons)
        chosen: List[Any] = [neurons[(start + i) % len(neurons)] for i in range(count)]
        ages: List[float] = []
        for n in chosen:
            info = selfattention.get_neuron_report(n)
            try:
                ages.append(float(info.get("age", 0.0)))
            except Exception:
                pass
        if not ages:
            return None
        avg = sum(ages) / len(ages)
        try:
            report("selfattention", "age_temperature", {"step": step_index, "avg": avg}, "events")
        except Exception:
            pass
        if avg < thr:
            selfattention.set_param("temperature", ytemp)
        else:
            selfattention.set_param("temperature", otemp)
        return None


__all__ = ["AgeTemperatureRoutine"]

