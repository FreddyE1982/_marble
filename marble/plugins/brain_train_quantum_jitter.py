"""QuantumJitterTrainPlugin jitters learning rate with a learnable factor.

The ``jitter_factor`` parameter, surfaced via ``expose_learnable_params``,
scales the incoming learning rate each walk, allowing the training process to
explore oscillating schedules far outside conventional optimisation
strategies.
"""

from __future__ import annotations

from typing import Any, Dict

from ..wanderer import expose_learnable_params
from ..reporter import report


@expose_learnable_params
def _jitter_param(wanderer, jitter_factor: float = 0.0):
    return jitter_factor


class QuantumJitterTrainPlugin:
    """Brain-train plugin applying a learnable jitter to the learning rate."""

    def before_walk(self, brain: "Brain", wanderer: "Wanderer", i: int) -> Dict[str, Any]:
        jit_t = _jitter_param(wanderer)
        try:
            jit = float(jit_t.detach().to("cpu").item())
        except Exception:
            jit = 0.0
        base_lr = float(getattr(wanderer, "lr", 0.0))
        lr = base_lr * (1.0 + jit)
        report("training", "quantum_jitter", {"walk": i, "lr": lr}, "brain")
        return {"lr": lr}


__all__ = ["QuantumJitterTrainPlugin"]

