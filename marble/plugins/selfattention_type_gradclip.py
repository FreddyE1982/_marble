from __future__ import annotations

from typing import Any, Dict, List

from ..wanderer import expose_learnable_params
from ..reporter import report


@expose_learnable_params
def _type_clip_params(
    wanderer,
    num_neurons: float = 3.0,
    start_index: float = 0.0,
    ratio_threshold: float = 0.5,
    max_norm: float = 1.0,
):
    return num_neurons, start_index, ratio_threshold, max_norm


class TypeGradClipRoutine:
    """Enable gradient clipping when diverse neuron types dominate."""

    def after_step(
        self,
        selfattention: "SelfAttention",
        reporter_ro: Any,
        wanderer: "Wanderer",
        step_index: int,
        ctx: Dict[str, Any],
    ):
        n_t, s_t, thr_t, m_t = _type_clip_params(wanderer)
        try:
            count = max(1, int(n_t.detach().to("cpu").item()))
            start = int(s_t.detach().to("cpu").item())
            thr = float(thr_t.detach().to("cpu").item())
            max_norm = float(m_t.detach().to("cpu").item())
        except Exception:
            count, start, thr, max_norm = 3, 0, 0.5, 1.0
        neurons = list(getattr(wanderer.brain, "neurons", {}).values())
        if not neurons:
            return None
        neurons.sort(key=lambda n: (getattr(n, "position", ()), id(n)))
        start = start % len(neurons)
        chosen: List[Any] = [neurons[(start + i) % len(neurons)] for i in range(count)]
        diverse = 0
        for n in chosen:
            info = selfattention.get_neuron_report(n)
            t = info.get("type_name")
            if t and t != "base":
                diverse += 1
        ratio = diverse / float(len(chosen))
        try:
            report("selfattention", "type_ratio", {"step": step_index, "ratio": ratio}, "events")
        except Exception:
            pass
        if ratio > thr:
            selfattention.set_param("_grad_clip", {"method": "norm", "max_norm": max_norm, "norm_type": 2.0})
        return None


__all__ = ["TypeGradClipRoutine"]

