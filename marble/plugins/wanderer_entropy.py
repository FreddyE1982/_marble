from __future__ import annotations

"""Entropy-aware Wanderer plugin.

Computes the entropy of choice weights and performs random exploration
while the entropy is below a learnable threshold.
"""

import math
import random
from typing import List, Tuple

from ..wanderer import expose_learnable_params


class EntropyAwarePlugin:
    """Explore until weight distribution has sufficient entropy."""

    @staticmethod
    @expose_learnable_params
    def _params(wanderer, *, entropy_threshold: float = 0.5):
        return (entropy_threshold,)

    def choose_next(self, wanderer, current, choices: List[Tuple["Synapse", str]]):
        if not choices:
            return None, "forward"
        (thr_t,) = self._params(wanderer)
        thr = float(thr_t.detach().to("cpu").item()) if hasattr(thr_t, "detach") else float(thr_t)
        weights = [max(float(getattr(c[0], "weight", 1.0)), 1e-6) for c in choices]
        total = sum(weights)
        probs = [w / total for w in weights]
        entropy = -sum(p * math.log(p) for p in probs)
        if entropy < thr:
            return random.choice(choices)
        return choices[probs.index(max(probs))]


__all__ = ["EntropyAwarePlugin"]
