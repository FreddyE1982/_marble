"""EntropyGateTrainPlugin perturbs walk length using a learnable gate.

The plugin introduces a ``entropy_gate`` parameter exposed via
``expose_learnable_params``. The parameter scales an entropy-like bonus to the
``max_steps`` for each walk, encouraging dynamic exploration of the graph.
"""

from __future__ import annotations

from typing import Any, Dict

from ..wanderer import expose_learnable_params
from ..reporter import report


@expose_learnable_params
def _entropy_param(wanderer, entropy_gate: float = 0.5):
    return entropy_gate


class EntropyGateTrainPlugin:
    """Brain-train plugin that alters walk steps based on a learnable gate."""

    def on_init(self, brain: "Brain", wanderer: "Wanderer", config: Dict[str, Any]) -> None:  # noqa: D401
        report("training", "entropy_gate_init", {}, "brain")

    def before_walk(self, brain: "Brain", wanderer: "Wanderer", i: int) -> Dict[str, Any]:
        gate_t = _entropy_param(wanderer)
        try:
            gate = float(gate_t.detach().to("cpu").item())
        except Exception:
            gate = 0.0
        bonus = int(abs(gate) * ((i % 3) + 1))
        report("training", "entropy_gate_before", {"walk": i, "bonus": bonus}, "brain")
        return {"max_steps": bonus + 1}


__all__ = ["EntropyGateTrainPlugin"]

