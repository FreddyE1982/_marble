from __future__ import annotations

from typing import Any, Dict

from ..reporter import report



class BaseNeuroplasticityPlugin:
    def on_init(self, wanderer: "Wanderer") -> None:
        try:
            report("neuroplasticity", "init", {"type": "base"}, "events")
        except Exception:
            pass

    def on_step(
        self,
        wanderer: "Wanderer",
        current: "Neuron",
        syn: "Synapse",
        direction: str,
        step_index: int,
        out_value: Any,
    ) -> None:
        try:
            report("neuroplasticity", "step", {"dir": direction, "step": step_index}, "events")
        except Exception:
            pass
        try:
            cfg = getattr(wanderer, "_neuro_cfg", {}) or {}
            grow_on_step = bool(cfg.get("grow_on_step_when_stuck", False))
            max_new = int(cfg.get("max_new_per_walk", 1))
            if not grow_on_step or getattr(current, "outgoing", None):
                return
            cur_new = int(wanderer._plugin_state.get("neuro_new_added", 0))
            if cur_new >= max_new:
                return
            brain = wanderer.brain
            try:
                avail = brain.available_indices()
            except Exception:
                avail = []
            if not avail:
                return
            last_pos = getattr(current, "position", None)
            chosen = None
            for cand in avail:
                try:
                    if cand != last_pos and brain.get_neuron(cand) is None:
                        chosen = cand
                        break
                except Exception:
                    continue
            if chosen is None:
                chosen = avail[0]
            try:
                brain.add_neuron(chosen, tensor=0.0)
                brain.connect(getattr(current, "position"), chosen, direction="uni")
                wanderer._plugin_state["neuro_new_added"] = cur_new + 1
                report("neuroplasticity", "grow_step", {"from": getattr(current, "position", None), "to": chosen}, "events")
            except Exception:
                pass
        except Exception:
            pass

    def on_walk_end(self, wanderer: "Wanderer", stats: Dict[str, Any]) -> None:
        try:
            visited = getattr(wanderer, "_visited", [])
            if not visited:
                return
            last = visited[-1]
            cfg = getattr(wanderer, "_neuro_cfg", {}) or {}
            grow_if_no_out = bool(cfg.get("grow_if_no_outgoing", True))
            max_new = int(cfg.get("max_new_per_walk", 1))
            enable_prune = bool(cfg.get("enable_prune", False))
            prune_gt = cfg.get("prune_if_outgoing_gt") if "prune_if_outgoing_gt" in cfg else None
            adjust_bias = cfg.get("adjust_bias_on_loss")

            if adjust_bias is not None:
                try:
                    delta = -float(stats.get("loss", 0.0))
                    last.bias = float(last.bias) + (
                        float(adjust_bias) * (1.0 if delta < 0 else -1.0)
                    )
                except Exception:
                    pass

            if enable_prune and prune_gt is not None:
                try:
                    if len(getattr(last, "outgoing", [])) > int(prune_gt):
                        syn = last.outgoing[-1]
                        wanderer.brain.remove_synapse(syn)
                        report(
                            "neuroplasticity",
                            "prune",
                            {"from": getattr(last, "position", None)},
                            "events",
                        )
                except Exception:
                    pass

            if getattr(last, "outgoing", None) and len(last.outgoing) > 0:
                return
            if not grow_if_no_out:
                return
            cur_new = int(wanderer._plugin_state.get("neuro_new_added", 0))
            if cur_new >= max_new:
                return
            brain = wanderer.brain
            try:
                avail = brain.available_indices()
            except Exception:
                avail = []
            if not avail:
                return
            last_pos = getattr(last, "position", None)
            chosen = None
            try:
                for cand in avail:
                    if cand != last_pos and brain.get_neuron(cand) is None:
                        chosen = cand
                        break
            except Exception:
                chosen = None
            if chosen is None:
                chosen = avail[0]
            try:
                brain.add_neuron(chosen, tensor=0.0)
            except Exception:
                pass
            try:
                brain.connect(getattr(last, "position"), chosen, direction="uni")
                wanderer._plugin_state["neuro_new_added"] = cur_new + 1
                report(
                    "neuroplasticity",
                    "grow",
                    {"from": getattr(last, "position", None), "to": chosen},
                    "events",
                )
            except Exception:
                pass
        except Exception:
            pass


__all__ = ["BaseNeuroplasticityPlugin"]

