from __future__ import annotations

"""Pheromone-trail inspired Wanderer plugin.

Maintains pheromone levels per synapse with learnable evaporation and
deposit rates, biasing traversal toward frequently used paths.
"""

from typing import Dict, List, Tuple

from ..wanderer import register_wanderer_type, expose_learnable_params


class PheromoneTrailPlugin:
    """Bias choices by accumulated pheromone levels."""

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer,
        *,
        pheromone_evap: float = 0.05,
        pheromone_deposit: float = 0.1,
    ):
        return (pheromone_evap, pheromone_deposit)

    def on_init(self, wanderer) -> None:
        wanderer._plugin_state["pheromones"] = {}

    def choose_next(self, wanderer, current, choices: List[Tuple["Synapse", str]]):
        if not choices:
            return None, "forward"
        state: Dict["Synapse", float] = wanderer._plugin_state.setdefault("pheromones", {})
        evap_t, depo_t = self._params(wanderer)
        evap = float(evap_t.detach().to("cpu").item()) if hasattr(evap_t, "detach") else float(evap_t)
        depo = float(depo_t.detach().to("cpu").item()) if hasattr(depo_t, "detach") else float(depo_t)
        for syn in list(state.keys()):
            state[syn] *= max(0.0, 1.0 - evap)
        def score(cd):
            syn = cd[0]
            return float(getattr(syn, "weight", 1.0)) * (1.0 + state.get(syn, 0.0))
        choice = max(choices, key=score)
        state[choice[0]] = state.get(choice[0], 0.0) + depo
        return choice


try:  # pragma: no cover
    register_wanderer_type("pheromone", PheromoneTrailPlugin())
except Exception:
    pass

__all__ = ["PheromoneTrailPlugin"]
