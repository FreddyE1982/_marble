"""SynapseBouncePlugin flips weight sign with a learnable bounce scale."""

from __future__ import annotations

from typing import Any

from ..wanderer import expose_learnable_params
from ..reporter import report


@expose_learnable_params
def _bounce_param(wanderer, bounce_scale: float = 1.0):
    return bounce_scale


class SynapseBouncePlugin:
    """Invert synapse weight sign based on a bounce coefficient."""

    def on_step(self, wanderer: "Wanderer", current: "Neuron", syn: "Synapse", direction: str, step_index: int, out_value: Any) -> None:
        bs_t = _bounce_param(wanderer)
        try:
            bs = float(bs_t.detach().to("cpu").item())
        except Exception:
            bs = 1.0
        try:
            if bs > 0.5:
                syn.weight = -float(getattr(syn, "weight", 0.0))
            report("neuroplasticity", "synapse_bounce", {"step": int(step_index), "scale": bs}, "plugins")
        except Exception:
            pass
        return None


__all__ = ["SynapseBouncePlugin"]

