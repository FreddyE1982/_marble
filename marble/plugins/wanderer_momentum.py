from __future__ import annotations

"""Momentum-biased Wanderer plugin.

Keeps an exponential moving average of previous synapse weights and adds
it as a bias term during selection. The momentum coefficient is exposed
through :func:`expose_learnable_params`.
"""

from typing import List, Tuple

from ..wanderer import register_wanderer_type, expose_learnable_params


class MomentumBiasPlugin:
    """Add momentum of previous weights to current scores."""

    @staticmethod
    @expose_learnable_params
    def _params(wanderer, *, momentum_coef: float = 0.9):
        return (momentum_coef,)

    def on_init(self, wanderer) -> None:
        wanderer._plugin_state["momentum"] = 0.0

    def choose_next(self, wanderer, current, choices: List[Tuple["Synapse", str]]):
        if not choices:
            return None, "forward"
        (coef_t,) = self._params(wanderer)
        coef = float(coef_t.detach().to("cpu").item()) if hasattr(coef_t, "detach") else float(coef_t)
        prev = wanderer._plugin_state.get("momentum", 0.0)
        def score(cd):
            return float(getattr(cd[0], "weight", 1.0)) + prev
        choice = max(choices, key=score)
        wanderer._plugin_state["momentum"] = coef * prev + float(getattr(choice[0], "weight", 1.0))
        return choice


try:  # pragma: no cover
    register_wanderer_type("momentum", MomentumBiasPlugin())
except Exception:
    pass

__all__ = ["MomentumBiasPlugin"]
