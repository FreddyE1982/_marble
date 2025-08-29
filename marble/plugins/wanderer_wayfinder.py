from __future__ import annotations

"""Wanderer plugin implementing navigation style path planning.

The plugin maintains a lightweight map of visited training states and uses a
heuristic search similar to A* to pick the next synapse.  To keep memory usage
bounded, rarely visited nodes are pruned.  Random exploration and periodic
map resets mitigate local optima.  Numerous behaviour weights are exposed as
Wanderer learnable parameters via :func:`expose_learnable_params`.
"""

from typing import Any, Dict, List, Tuple, Optional
import random

from ..wanderer import register_wanderer_type, expose_learnable_params
from ..reporter import report


class WayfinderPlugin:
    """Navigation-inspired path planner for :class:`~marble.wanderer.Wanderer`.

    Per-wanderer maps store node visit counts and edge costs.  On each
    ``choose_next`` call the plugin performs a small heuristic search favouring
    low cost edges while occasionally exploring random alternatives.  Maps are
    pruned to avoid unbounded growth and periodically cleared to trigger
    replanning.
    """

    def __init__(self) -> None:
        # Per-wanderer state indexed by ``id(wanderer)``
        self._maps: Dict[int, Dict[int, Dict[str, Any]]] = {}
        self._last_node: Dict[int, int] = {}
        self._steps: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Learnable and configurable parameters
    # ------------------------------------------------------------------
    @staticmethod
    @expose_learnable_params
    def _get_params(
        wanderer: "Wanderer",
        *,
        cost_weight: float = 1.0,
        visit_penalty: float = 0.1,
        explore_prob: float = 0.05,
        prune_ratio: float = 0.5,
        heuristic_bias: float = 0.0,
        replan_interval: float = 10.0,
    ):
        """Return tensors for all tunable parameters.

        Parameters
        ----------
        cost_weight: multiplies synapse weight when evaluating edges.
        visit_penalty: additional cost for frequently visited nodes.
        explore_prob: probability to ignore planning and pick a random edge.
        prune_ratio: fraction of nodes to drop when the map exceeds capacity.
        heuristic_bias: constant bias added to each evaluated edge.
        replan_interval: number of steps between automatic map resets.
        """

        return (
            cost_weight,
            visit_penalty,
            explore_prob,
            prune_ratio,
            heuristic_bias,
            replan_interval,
        )

    def _params(self, wanderer: "Wanderer"):
        return WayfinderPlugin._get_params(wanderer)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _get_map(self, wanderer: "Wanderer") -> Dict[int, Dict[str, Any]]:
        return self._maps.setdefault(id(wanderer), {})

    def _prune(
        self, mp: Dict[int, Dict[str, Any]], prune_ratio: float, max_nodes: int
    ) -> None:
        if len(mp) <= max_nodes:
            return
        # Drop least visited nodes according to prune_ratio
        cut = max(0, int(len(mp) * prune_ratio))
        if cut == 0:
            return
        sorted_nodes = sorted(mp.items(), key=lambda kv: kv[1].get("visits", 0))
        for nid, _ in sorted_nodes[:cut]:
            mp.pop(nid, None)

    def _a_star(
        self,
        current: "Neuron",
        choices: List[Tuple["Synapse", str]],
        mp: Dict[int, Dict[str, Any]],
        cost_w: float,
        visit_pen: float,
        bias: float,
    ) -> Tuple["Synapse", str]:
        # Single-step heuristic evaluation
        best = choices[0]
        best_score = float("inf")
        for syn, direction in choices:
            nxt = syn.target if direction == "forward" else syn.source
            nid = id(nxt)
            visits = mp.get(nid, {}).get("visits", 0)
            weight = float(getattr(syn, "weight", 1.0))
            score = cost_w * weight + visit_pen * visits + bias
            if score < best_score:
                best_score = score
                best = (syn, direction)
        return best

    # ------------------------------------------------------------------
    # Wanderer plugin interface
    # ------------------------------------------------------------------
    def choose_next(
        self,
        wanderer: "Wanderer",
        current: "Neuron",
        choices: List[Tuple["Synapse", str]],
    ):
        if not choices:
            return None, "forward"

        (
            cost_w,
            visit_pen,
            explore_prob,
            prune_ratio,
            bias,
            replan_int,
        ) = self._params(wanderer)

        torch = getattr(wanderer, "_torch", None)
        to_cpu = (
            lambda t: float(t.detach().to("cpu").item()) if torch is not None else float(t)
        )

        cost_w_f = to_cpu(cost_w)
        visit_pen_f = to_cpu(visit_pen)
        explore_prob_f = to_cpu(explore_prob)
        prune_ratio_f = to_cpu(prune_ratio)
        bias_f = to_cpu(bias)
        replan_int_f = max(1, int(to_cpu(replan_int)))

        mp = self._get_map(wanderer)
        w_id = id(wanderer)
        step = self._steps.get(w_id, 0) + 1
        self._steps[w_id] = step

        # Periodic replanning by clearing the map
        if step % replan_int_f == 0:
            mp.clear()

        max_nodes = int(getattr(wanderer, "_neuro_cfg", {}).get("wayfinder_max_nodes", 100))
        self._prune(mp, prune_ratio_f, max_nodes)

        if random.random() < explore_prob_f:
            return random.choice(choices)

        return self._a_star(current, choices, mp, cost_w_f, visit_pen_f, bias_f)

    def on_step(
        self,
        wanderer: "Wanderer",
        current: "Neuron",
        next_syn: Optional["Synapse"],
        direction: str,
        step_index: int,
        out_value: Any,
    ) -> None:
        w_id = id(wanderer)
        mp = self._get_map(wanderer)
        prev_id = self._last_node.get(w_id, id(current))
        if next_syn is not None:
            nxt = next_syn.target if direction == "forward" else next_syn.source
        else:
            nxt = current
        next_id = id(nxt)
        node = mp.setdefault(prev_id, {"visits": 0, "edges": {}})
        node["visits"] = node.get("visits", 0) + 1

        # Edge cost uses current step loss if available
        loss_v = 0.0
        try:
            cur_loss_t = wanderer._walk_ctx.get("cur_loss_tensor")  # type: ignore[attr-defined]
            if cur_loss_t is not None:
                torch = getattr(wanderer, "_torch", None)
                if torch is not None:
                    loss_v = float(cur_loss_t.detach().to("cpu").item())
        except Exception:
            loss_v = 0.0
        node["edges"][next_id] = loss_v
        self._last_node[w_id] = next_id

        try:
            report(
                "wanderer",
                "wayfinder_step",
                {"from": prev_id, "to": next_id, "loss": loss_v},
                "plugins",
            )
        except Exception:
            pass

    def on_walk_end(self, wanderer: "Wanderer") -> None:
        w_id = id(wanderer)
        self._last_node.pop(w_id, None)


try:  # pragma: no cover - registration should not break tests on import
    register_wanderer_type("wayfinder", WayfinderPlugin())
except Exception:
    pass

__all__ = ["WayfinderPlugin"]

