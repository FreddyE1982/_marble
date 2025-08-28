from __future__ import annotations

from typing import List, Tuple

from ..wanderer import register_wanderer_type


class EpsilonGreedyChooserPlugin:
    """Epsilon-greedy chooser: with probability epsilon pick a random choice; otherwise prefer by synapse.weight.

    epsilon is read from `wanderer._neuro_cfg.get('epsilongreedy_epsilon', 0.1)`.
    """

    def choose_next(self, wanderer: "Wanderer", current: "Neuron", choices: List[Tuple["Synapse", str]]):
        if not choices:
            return None, "forward"
        import random as _r
        cfg = getattr(wanderer, "_neuro_cfg", {}) or {}
        eps = float(cfg.get("epsilongreedy_epsilon", 0.1))
        if _r.random() < eps:
            return _r.choice(choices)
        # fallback to highest weight
        return max(choices, key=lambda cd: float(getattr(cd[0], "weight", 1.0)))


try:
    register_wanderer_type("epsilongreedy", EpsilonGreedyChooserPlugin())
except Exception:
    pass

__all__ = ["EpsilonGreedyChooserPlugin"]
