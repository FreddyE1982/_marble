from __future__ import annotations

"""Wanderer plugin that auto-defines brain lobes via a learnable threshold.

The plugin inspects neuron positions on each walk and splits them into two
lobes along the first coordinate axis.  The split location is governed by a
single learnable parameter exposed through :func:`expose_learnable_params`.
This makes the lobe boundary differentiable so training can adapt where the
brain is partitioned.
"""

from typing import List

from ..wanderer import expose_learnable_params


class AutoLobePlugin:
    """Automatically create two lobes based on neuron positions."""

    @staticmethod
    @expose_learnable_params
    def _get_params(wanderer: "Wanderer", *, autolobe_threshold: float = 0.5):
        """Return the learnable threshold separating both lobes."""

        return autolobe_threshold

    def _threshold(self, wanderer: "Wanderer") -> float:
        thr = AutoLobePlugin._get_params(wanderer)
        torch = getattr(wanderer, "_torch", None)
        if torch is not None and hasattr(thr, "detach"):
            return float(thr.detach().to("cpu").item())
        return float(thr)

    def before_walk(self, wanderer: "Wanderer", start: "Neuron") -> None:
        """Ensure two lobes exist before each walk."""

        brain = wanderer.brain
        thr = self._threshold(wanderer)
        low: List["Neuron"] = []
        high: List["Neuron"] = []
        for n in brain.neurons.values():
            pos = getattr(n, "position", (0,))
            axis0 = pos[0] if isinstance(pos, tuple) else pos
            (low if axis0 <= thr else high).append(n)
        if low:
            brain.define_lobe("autolobe_low", low)
        if high:
            brain.define_lobe("autolobe_high", high)


__all__ = ["AutoLobePlugin"]

