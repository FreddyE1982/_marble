from __future__ import annotations

"""Boltzmann-style exploration for the Wanderer.

Chooses the next synapse based on a softmax over weights using a
learnable temperature parameter exposed through
:func:`expose_learnable_params`.
"""

from typing import List, Tuple
import math
import random

from ..wanderer import expose_learnable_params


class BoltzmannChooserPlugin:
    """Select synapses via Boltzmann exploration."""

    @staticmethod
    @expose_learnable_params
    def _params(wanderer, *, temperature: float = 1.0):
        return (temperature,)

    def choose_next(self, wanderer, current, choices: List[Tuple["Synapse", str]]):
        if not choices:
            return None, "forward"
        (temp_t,) = self._params(wanderer)
        temp = float(getattr(temp_t, "detach", lambda: temp_t)().to("cpu").item()) if hasattr(temp_t, "detach") else float(temp_t)
        torch = getattr(wanderer, "_torch", None)
        if torch is not None:
            weights = torch.tensor(
                [float(getattr(c[0], "weight", 1.0)) for c in choices],
                dtype=torch.float32,
                device=getattr(wanderer, "_device", "cpu"),
            )
            probs = torch.softmax(weights / temp, dim=0)
            idx = int(torch.multinomial(probs, 1).item())
            return choices[idx]
        weights = [float(getattr(c[0], "weight", 1.0)) for c in choices]
        exps = [math.exp(w / temp) for w in weights]
        total = sum(exps)
        r = random.random() * total
        upto = 0.0
        for e, choice in zip(exps, choices):
            upto += e
            if upto >= r:
                return choice
        return choices[-1]


__all__ = ["BoltzmannChooserPlugin"]
