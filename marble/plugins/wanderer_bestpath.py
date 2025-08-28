from __future__ import annotations

from typing import Any, List, Tuple

from ..reporter import report


class BestLossPathPlugin:
    """On first choose_next in a walk, search paths up to max_steps and upweight best path synapses.

    The synapse weights are increased along the best path so that a weights-driven
    plugin (e.g., WanderAlongSynapseWeightsPlugin) will prefer it.
    """

    def _simulate(self, wanderer: "Wanderer", start: "Neuron", max_steps: int):
        # DFS over choices; returns (best_loss, best_path_edges)
        best = (float("inf"), [])

        def rec(node: "Neuron", carried, depth: int, outputs: List[Any], edges: List[Tuple["Synapse", str]]):
            nonlocal best
            if depth >= max_steps:
                # Evaluate
                loss_t = wanderer._compute_loss(outputs)
                try:
                    loss_v = float(loss_t.detach().to("cpu").item())
                except Exception:
                    loss_v = float("inf")
                if loss_v < best[0]:
                    best = (loss_v, list(edges))
                return
            # Compute current output
            out = node.forward(carried)
            outputs2 = outputs + [out]
            # Choices from node
            choices = wanderer._gather_choices(node)
            if not choices:
                loss_t = wanderer._compute_loss(outputs2)
                try:
                    loss_v = float(loss_t.detach().to("cpu").item())
                except Exception:
                    loss_v = float("inf")
                if loss_v < best[0]:
                    best = (loss_v, list(edges))
                return
            for syn, dir_str in choices:
                # Determine next node
                nxt = syn.target if dir_str == "forward" else syn.source
                rec(nxt, out, depth + 1, outputs2, edges + [(syn, dir_str)])

        rec(start, None, 0, [], [])
        return best

    def _bump_weights(self, path_edges: List[Tuple["Synapse", str]], factor: float = 1.0, add: float = 1.0):
        for syn, _ in path_edges:
            try:
                w = float(getattr(syn, "weight", 1.0))
                syn.weight = w * factor + add
            except Exception:
                pass

    def choose_next(self, wanderer: "Wanderer", current: "Neuron", choices: List[Tuple["Synapse", str]]):
        # Run once per walk
        if not getattr(wanderer, "_plugin_state", None):
            wanderer._plugin_state = {}
        if not wanderer._plugin_state.get("bestlosspath_applied"):
            # Read config from wanderer._neuro_cfg
            cfg = getattr(wanderer, "_neuro_cfg", {}) or {}
            max_steps_cfg = int(cfg.get("bestlosspath_search_steps", 3))
            bump_factor = float(cfg.get("bestlosspath_bump_factor", 1.0))
            bump_add = float(cfg.get("bestlosspath_bump_add", 1.0))
            best_loss, best_path = self._simulate(wanderer, current, max_steps_cfg)
            self._bump_weights(best_path, factor=bump_factor, add=bump_add)
            wanderer._plugin_state["bestlosspath_applied"] = True
            try:
                nodes = []
                for syn, d in best_path:
                    try:
                        nodes.append({
                            "src": getattr(getattr(syn, "source", None), "position", None),
                            "dst": getattr(getattr(syn, "target", None), "position", None),
                        })
                    except Exception:
                        pass
                report("wanderer", "bestlosspath", {"path": nodes, "loss": best_loss}, "plugins")
            except Exception:
                pass
        return None, "forward"

__all__ = ["BestLossPathPlugin"]
