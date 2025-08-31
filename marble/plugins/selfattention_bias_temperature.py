from __future__ import annotations

from typing import Any, Dict, List

from ..wanderer import expose_learnable_params
from ..reporter import report
from .selfattention_metric_utils import metric_factor


@expose_learnable_params
def _bias_temp_params(
    wanderer,
    num_neurons: float = 3.0,
    start_index: float = 0.0,
    bias_threshold: float = 0.0,
    high_temp: float = 2.0,
    low_temp: float = 0.5,
):
    return num_neurons, start_index, bias_threshold, high_temp, low_temp


class BiasTemperatureRoutine:
    """Adjust attention temperature based on average neuron bias."""

    def after_step(
        self,
        selfattention: "SelfAttention",
        reporter_ro: Any,
        wanderer: "Wanderer",
        step_index: int,
        ctx: Dict[str, Any],
    ):
        n_t, s_t, thr_t, hi_t, lo_t = _bias_temp_params(wanderer)
        try:
            count = max(1, int(n_t.detach().to("cpu").item()))
            start = int(s_t.detach().to("cpu").item())
            thr = float(thr_t.detach().to("cpu").item())
            hi = float(hi_t.detach().to("cpu").item())
            lo = float(lo_t.detach().to("cpu").item())
        except Exception:
            count, start, thr, hi, lo = 3, 0, 0.0, 2.0, 0.5
        factor = metric_factor(ctx, "bias_temperature")
        thr *= 1.0 + factor
        hi *= factor
        lo *= factor
        neurons = list(getattr(wanderer.brain, "neurons", {}).values())
        if not neurons:
            return None
        neurons.sort(key=lambda n: (getattr(n, "position", ()), id(n)))
        start = start % len(neurons)
        chosen: List[Any] = [neurons[(start + i) % len(neurons)] for i in range(count)]
        biases: List[float] = []
        for n in chosen:
            info = selfattention.get_neuron_report(n)
            try:
                biases.append(float(info.get("bias", 0.0)))
            except Exception:
                pass
        if not biases:
            return None
        avg = sum(biases) / len(biases)
        try:
            report("selfattention", "bias_temperature", {"step": step_index, "avg": avg}, "events")
        except Exception:
            pass
        if avg > thr:
            selfattention.set_param("temperature", hi)
        else:
            selfattention.set_param("temperature", lo)
        return None


__all__ = ["BiasTemperatureRoutine"]

