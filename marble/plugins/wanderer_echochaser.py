"""Wanderer plugin that chases echoes of prior choices."""

from __future__ import annotations

from typing import List, Tuple

import math
import random

from ..reporter import report
from ..wanderer import expose_learnable_params


class EchoChaserPlugin:
    """Reinforces frequently chosen synapses with an exponential gain."""

    @staticmethod
    @expose_learnable_params
    def _params(wanderer, *, echo_gain: float = 1.0):
        return (echo_gain,)

    def choose_next(self, wanderer, current, choices: List[Tuple["Synapse", str]]):
        if not choices:
            return None, "forward"
        state = wanderer._plugin_state.setdefault("echo_counts", {})
        (gain_t,) = self._params(wanderer)
        torch = getattr(wanderer, "_torch", None)
        device = getattr(wanderer, "_device", "cpu")
        if torch is not None:
            counts = torch.tensor(
                [state.get(id(c[0]), 0) for c in choices],
                dtype=torch.float32,
                device=device,
            )
            weights = torch.exp(counts * gain_t)
            probs = weights / torch.sum(weights)
            idx = int(torch.multinomial(probs, 1).item())
        else:
            weights = [math.exp(state.get(id(c[0]), 0) * float(gain_t)) for c in choices]
            total = sum(weights)
            r = random.random() * total
            upto = 0.0
            idx = 0
            for i, w in enumerate(weights):
                upto += w
                if upto >= r:
                    idx = i
                    break
        syn, direction = choices[idx]
        state[id(syn)] = state.get(id(syn), 0) + 1
        report("wanderer", "echo_chaser", {"choice": idx}, "plugins")
        return syn, direction


__all__ = ["EchoChaserPlugin"]

