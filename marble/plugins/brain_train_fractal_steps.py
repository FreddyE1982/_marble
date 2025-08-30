"""FractalStepsTrainPlugin derives step counts from a learnable fractal dim.

By exposing ``fractal_dim`` the plugin lets training walks expand or contract
their length according to a pseudo-fractal scaling, probing erratic pacing
schemes that classical curricula ignore.
"""

from __future__ import annotations

from typing import Any, Dict

from ..wanderer import expose_learnable_params
from ..reporter import report


@expose_learnable_params
def _fract_param(wanderer, fractal_dim: float = 1.0):
    return fractal_dim


class FractalStepsTrainPlugin:
    """Brain-train plugin computing steps from a learnable fractal dimension."""

    def before_walk(self, brain: "Brain", wanderer: "Wanderer", i: int) -> Dict[str, Any]:
        fract_t = _fract_param(wanderer)
        try:
            fract = float(fract_t.detach().to("cpu").item())
        except Exception:
            fract = 1.0
        steps = int(abs(fract) * (i + 1)) + 1
        report("training", "fractal_steps", {"walk": i, "steps": steps}, "brain")
        return {"max_steps": steps}


__all__ = ["FractalStepsTrainPlugin"]

