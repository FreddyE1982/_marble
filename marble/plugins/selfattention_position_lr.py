from __future__ import annotations

from typing import Any, Dict, List
import math

from ..wanderer import expose_learnable_params
from ..reporter import report


@expose_learnable_params
def _pos_lr_params(
    wanderer,
    num_neurons: float = 3.0,
    start_index: float = 0.0,
    scale: float = 1.0,
):
    return num_neurons, start_index, scale


class PositionLRRoutine:
    """Scale learning rate based on average neuron position norm."""

    def after_step(
        self,
        selfattention: "SelfAttention",
        reporter_ro: Any,
        wanderer: "Wanderer",
        step_index: int,
        ctx: Dict[str, Any],
    ):
        n_t, s_t, sc_t = _pos_lr_params(wanderer)
        try:
            count = max(1, int(n_t.detach().to("cpu").item()))
            start = int(s_t.detach().to("cpu").item())
            scale = float(sc_t.detach().to("cpu").item())
        except Exception:
            count, start, scale = 3, 0, 1.0
        neurons = list(getattr(wanderer.brain, "neurons", {}).values())
        if not neurons:
            return None
        neurons.sort(key=lambda n: (getattr(n, "position", ()), id(n)))
        start = start % len(neurons)
        chosen: List[Any] = [neurons[(start + i) % len(neurons)] for i in range(count)]
        norms: List[float] = []
        for n in chosen:
            info = selfattention.get_neuron_report(n)
            pos = info.get("position")
            if isinstance(pos, (list, tuple)):
                try:
                    norms.append(math.sqrt(sum(float(x) ** 2 for x in pos)))
                except Exception:
                    pass
        if not norms:
            return None
        avg = sum(norms) / len(norms)
        try:
            report("selfattention", "position_lr", {"step": step_index, "avg": avg}, "events")
        except Exception:
            pass
        cur_lr = float(getattr(wanderer, "current_lr", 0.01) or 0.01)
        new_lr = cur_lr / (1.0 + scale * avg)
        selfattention.set_param("lr_override", new_lr)
        return None


__all__ = ["PositionLRRoutine"]

