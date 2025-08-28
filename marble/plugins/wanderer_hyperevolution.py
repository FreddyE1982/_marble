from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..wanderer import WANDERER_TYPES_REGISTRY as _WANDERER_TYPES, NEURO_TYPES_REGISTRY as _NEURO_TYPES
from ..marblemain import _PARADIGM_TYPES


class HyperEvolutionPlugin:
    """Evolutionary architecture search over plugin stacks and parameters without cloning the brain.

    Modes (via `wanderer._neuro_cfg['hyper_evo_mode']`):
    - "per_walk" (default): before the first training walk, run `hyper_evo_steps` evolution steps
      using temporary changes and full rollback per step. After finishing, configure the best
      architecture, deactivate this plugin, and proceed with normal training.
    - "per_step": at the beginning of each training step, run `hyper_evo_steps` evolution steps,
      configure the best architecture so far, then continue the step. This lets the search continue
      during training.

    Config (via `wanderer._neuro_cfg`):
      - hyper_evo_steps (int, default 50): evolution steps per run
      - hyper_eval_steps (int, default 3): bounded evaluation walk steps (with lr=0)
      - hyper_evo_mode (str, default "per_walk"): "per_walk" | "per_step"

    This search mutates across all registered plugin families (wanderer, neuroplasticity, paradigms)
    and any numeric parameters present in `wanderer._neuro_cfg`. No plugins or parameters are excluded.
    Each evolution step applies a temporary mutation, performs a short evaluation walk (lr=0) to get
    loss and speed metrics, and then rolls back every change unless both metrics improved; accepted
    mutations are accumulated into the current best genome. At the end of the run, the best genome is
    applied persistently (no temporary stacks), and this plugin deactivates itself in per_walk mode.
    """

    def on_init(self, wanderer: "Wanderer") -> None:
        state = getattr(wanderer, "_plugin_state", None)
        if state is None:
            wanderer._plugin_state = {}
            state = wanderer._plugin_state
        state.setdefault("hyper", {
            "gen": 0,
            "pop": [],  # legacy fields retained (not used by new evo loop)
            "idx": 0,
            "last_handle": None,
            "last_paradigms": None,
            "best": None,
            "best_score": None,
            "best_loss": None,
            "best_speed": None,
            "done": False,
            "disabled": False,
        })
        if not state["hyper"]["pop"]:
            self._init_population(wanderer)

    def _init_population(self, wanderer: "Wanderer") -> None:
        cfg = getattr(wanderer, "_neuro_cfg", {}) or {}
        pop_size = int(cfg.get("hyper_pop", 8))
        import random as _r
        pop = []
        wnames = list(_WANDERER_TYPES.keys())
        nnames = list(_NEURO_TYPES.keys())
        pnames = list(_PARADIGM_TYPES.keys()) if '_PARADIGM_TYPES' in globals() else []
        for _ in range(pop_size):
            genome = {
                "w": _r.sample(wnames, k=min(len(wnames), _r.randint(0, min(3, len(wnames))))) if wnames else [],
                "n": _r.sample(nnames, k=min(len(nnames), _r.randint(0, min(2, len(nnames))))) if nnames else [],
                "p": _r.sample(pnames, k=min(len(pnames), _r.randint(0, min(2, len(pnames))))) if pnames else [],
                "cfg": {},
                "score": None,
            }
            # Ensure at least one wanderer plugin is present to make evolution impactful.
            if not genome["w"] and wnames:
                try:
                    genome["w"] = [_r.choice(wnames)]
                except Exception:
                    pass
            # Encourage at least one paradigm to be active for diversity
            if not genome["p"] and pnames:
                try:
                    genome["p"] = [_r.choice(pnames)]
                except Exception:
                    pass
            pop.append(genome)
        # Seed population with a strong, commonly useful combination if available
        try:
            if wnames and ("bestlosspath" in wnames) and ("wanderalongsynapseweights" in wnames):
                pop[0] = {
                    "w": ["bestlosspath", "wanderalongsynapseweights"],
                    "n": [],
                    "p": [],
                    "cfg": {},
                    "score": None,
                }
        except Exception:
            pass


        wanderer._plugin_state["hyper"]["pop"] = pop
        wanderer._plugin_state["hyper"]["idx"] = 0

    def before_walk(self, wanderer: "Wanderer", start: "Neuron") -> None:
        # Run evolution per mode and apply the best architecture if needed
        st = wanderer._plugin_state["hyper"]
        if st.get("disabled"):
            return
        cfg = getattr(wanderer, "_neuro_cfg", {}) or {}
        mode = str(cfg.get("hyper_evo_mode", "per_walk")).lower()
        steps = int(cfg.get("hyper_evo_steps", 50))
        if mode == "per_walk" and not st.get("done"):
            self._run_evolution(wanderer, start, steps)
            self._apply_best_persistently(wanderer)
            st["done"] = True
            st["disabled"] = True
            return

        # Legacy selection/apply path retained for compatibility
        pop = st["pop"]
        if not pop:
            self._init_population(wanderer)
            pop = st["pop"]
        # Selection strategy: exploit best-so-far or explore via tournament
        import random as _r
        idx = 0
        try:
            # Identify scored genomes
            scored = [(i, float(g.get("score_avg", g.get("score", float("inf"))))) for i, g in enumerate(pop)]
            scored = [(i, s) for (i, s) in scored if s != float("inf")]
            best_idx = st.get("best_idx")
            if scored:
                # Track current best
                scored.sort(key=lambda t: t[1])
                current_best_idx = scored[0][0]
                st["best_idx"] = current_best_idx
                # Epsilon-greedy: mostly exploit best, sometimes explore
                eps = 0.3
                if best_idx is not None and _r.random() > eps:
                    idx = int(best_idx)
                else:
                    # Tournament among a random subset
                    k = max(2, min(5, len(pop)))
                    cand = _r.sample(range(len(pop)), k=k)
                    # Rank by available scores, unknown treated as mid-rank
                    def score_of(i):
                        g = pop[i]
                        s = g.get("score_avg", g.get("score"))
                        return float(s) if s is not None else float("inf")
                    idx = min(cand, key=score_of)
            else:
                # No scores yet: random selection
                idx = _r.randrange(0, len(pop))
        except Exception:
            idx = 0
        st["idx"] = idx
        genome = pop[idx]
        # Apply stacks
        handle = push_temporary_plugins(wanderer, wanderer_types=genome.get("w"), neuro_types=genome.get("n"))
        st["last_handle"] = handle
        # Toggle paradigms: enable selected, disable others previously managed
        prev = st.get("last_paradigms")
        act = []
        try:
            # disable previous
            if prev:
                for nm in prev:
                    try:
                        wanderer.brain.enable_paradigm(nm, enabled=False)
                    except Exception:
                        pass
            for nm in genome.get("p", []) or []:
                try:
                    wanderer.brain.enable_paradigm(nm, enabled=True)
                    act.append(nm)
                except Exception:
                    pass
            st["last_paradigms"] = act
        except Exception:
            pass
        # Snapshot current neuro config to isolate genome effects
        try:
            st["cfg_prev"] = dict(getattr(wanderer, "_neuro_cfg", {}) or {})
        except Exception:
            st["cfg_prev"] = None
        # Snapshot current LR override if any
        try:
            st["lr_prev"] = getattr(wanderer, "lr_override", None)
        except Exception:
            st["lr_prev"] = None
        # Merge any genome cfg params into wanderer._neuro_cfg (restored on walk end)
        try:
            for k, v in (genome.get("cfg") or {}).items():
                wanderer._neuro_cfg[k] = v
        except Exception:
            pass
        # If genome proposes an LR override, apply it (treated like any other evolvable param)
        try:
            lr_prop = (genome.get("cfg") or {}).get("lr_override", None)
            if lr_prop is not None:
                try:
                    wanderer.lr_override = float(lr_prop)
                except Exception:
                    pass
        except Exception:
            pass

    def on_step(self, wanderer: "Wanderer", current: "Neuron", syn: "Synapse", direction: str, step_index: int, out_value: Any) -> None:
        # Per-step mode: run micro evolution before continuing the step
        st = wanderer._plugin_state.get("hyper", {})
        if st.get("disabled"):
            return
        cfg = getattr(wanderer, "_neuro_cfg", {}) or {}
        mode = str(cfg.get("hyper_evo_mode", "per_walk")).lower()
        if mode != "per_step":
            return
        steps = int(cfg.get("hyper_evo_steps", 50))
        self._run_evolution(wanderer, current, steps)
        self._apply_best_persistently(wanderer)
        return

    def on_walk_end(self, wanderer: "Wanderer", stats: Dict[str, Any]) -> None:
        # Evaluate fitness
        st = wanderer._plugin_state.get("hyper", {})
        pop = st.get("pop", [])
        idx = st.get("idx", 0) % (len(pop) if pop else 1)
        if not pop:
            return
        genome = pop[idx]
        # Compute objectives (used only for legacy population mode)
        loss = float(stats.get("loss", 0.0))
        steps = int(stats.get("steps", 0))
        sm = stats.get("step_metrics", []) or []
        # per-step time
        dts = [m.get("dt") for m in sm if m.get("dt") is not None]
        mean_dt = (sum(dts) / len(dts)) if dts else 0.0
        # loss decrease speed (positive good): use -avg(delta) truncated at 0
        deltas = [float(m.get("delta", 0.0)) for m in sm]
        if deltas:
            avg_delta = sum(deltas) / len(deltas)
            speed = max(0.0, -avg_delta)
        else:
            speed = 0.0
        # brain size
        try:
            brain_size = len(getattr(wanderer.brain, "neurons", {}))
        except Exception:
            brain_size = 0
        # Score: minimize loss, mean_dt, brain_size; maximize speed => use reciprocal
        score = (loss) + (mean_dt) + (brain_size * 1e-3) + (1.0 / (1e-6 + speed) if speed > 0 else 1.0)
        # Update instantaneous and running-average scores for the genome
        genome["score"] = float(score)
        try:
            prev_avg = float(genome.get("score_avg", score))
            trials = int(genome.get("trials", 0)) + 1
            new_avg = (prev_avg * (trials - 1) + float(score)) / max(1, trials)
            genome["score_avg"] = float(new_avg)
            genome["trials"] = trials
        except Exception:
            genome["score_avg"] = float(score)
            genome["trials"] = 1
        # Restore stacks
        try:
            if st.get("last_handle") is not None:
                pop_handle = st["last_handle"]
                pop_temporary_plugins(wanderer, pop_handle)
                st["last_handle"] = None
        except Exception:
            pass
        # Restore neuro config snapshot
        try:
            prev = st.get("cfg_prev", None)
            if prev is not None:
                wanderer._neuro_cfg = dict(prev)
                st["cfg_prev"] = None
        except Exception:
            pass
        # Restore LR override
        try:
            if "lr_prev" in st:
                setattr(wanderer, "lr_override", st.get("lr_prev", None))
                st["lr_prev"] = None
        except Exception:
            pass
        # Harvest observed neuro_cfg keys to expand future mutation search space
        try:
            kb = st.setdefault("key_bank", set())
            for k in (getattr(wanderer, "_neuro_cfg", {}) or {}).keys():
                try:
                    kb.add(str(k))
                except Exception:
                    pass
        except Exception:
            pass

        # Advance and evolve at end of a population cycle (count-based, independent of chosen indices)
        st["eval_count"] = int(st.get("eval_count", 0)) + 1
        if st["eval_count"] >= max(1, len(pop)):
            st["eval_count"] = 0
            self._evolve(wanderer)

    def _evolve(self, wanderer: "Wanderer") -> None:
        import random as _r
        cfg = getattr(wanderer, "_neuro_cfg", {}) or {}
        # Adaptive mutation rate around configured baseline
        base_mut = float(cfg.get("hyper_mut", 0.3))
        st = wanderer._plugin_state["hyper"]
        cur_mut = float(st.get("mut_rate", base_mut))
        mut_rate = max(0.05, min(0.8, cur_mut))
        keep = int(cfg.get("hyper_keep", 2))
        pop = st["pop"]
        # Sort by running-average score ascending when available (else instantaneous)
        pop.sort(key=lambda g: float(g.get("score_avg", g.get("score", float("inf")))))
        keepers = pop[:max(1, min(keep, len(pop)))]
        # Track best score to adapt mutation rate
        try:
            best_now = float(keepers[0].get("score_avg", keepers[0].get("score", float("inf")))) if keepers else float("inf")
            last_best = float(st.get("last_best", float("inf")))
            if best_now < last_best:
                st["mut_rate"] = max(0.05, mut_rate * 0.9)
            else:
                st["mut_rate"] = min(0.8, mut_rate * 1.1)
            st["last_best"] = best_now
        except Exception:
            st["mut_rate"] = mut_rate
        # Refill
        wnames = list(_WANDERER_TYPES.keys())
        nnames = list(_NEURO_TYPES.keys())
        pnames = list(_PARADIGM_TYPES.keys()) if '_PARADIGM_TYPES' in globals() else []
        new_pop = []
        for g in keepers:
            new_pop.append({"w": list(g["w"]), "n": list(g["n"]), "p": list(g["p"]), "cfg": dict(g.get("cfg") or {}), "score": None, "score_avg": g.get("score_avg"), "trials": g.get("trials")})
        while len(new_pop) < len(pop):
            # Crossover (pick two parents)
            a, b = _r.sample(keepers, k=2) if len(keepers) >= 2 else (keepers[0], keepers[0])
            child = {
                "w": list(set(_r.sample(a["w"] + b["w"], k=min(len(a["w"] + b["w"]), _r.randint(0, min(3, len(wnames))))))) if (a["w"] or b["w"]) else [],
                "n": list(set(_r.sample(a["n"] + b["n"], k=min(len(a["n"] + b["n"]), _r.randint(0, min(2, len(nnames))))))) if (a["n"] or b["n"]) else [],
                "p": list(set(_r.sample(a["p"] + b["p"], k=min(len(a["p"] + b["p"]), _r.randint(0, min(2, len(pnames))))))) if (a["p"] or b["p"]) else [],
                "cfg": dict(a.get("cfg") or {}),
                "score": None,
            }
            # Mutate types
            if _r.random() < mut_rate and wnames:
                # flip one wanderer plugin
                choice = _r.choice(wnames)
                if choice in child["w"]:
                    child["w"].remove(choice)
                else:
                    child["w"].append(choice)
            if _r.random() < mut_rate and nnames:
                choice = _r.choice(nnames)
                if choice in child["n"]:
                    child["n"].remove(choice)
                else:
                    child["n"].append(choice)
            if _r.random() < mut_rate and pnames:
                choice = _r.choice(pnames)
                if choice in child["p"]:
                    child["p"].remove(choice)
                else:
                    child["p"].append(choice)
        # Mutate generic numeric cfg entries over a broad key bank
        keys = set((child.get("cfg") or {}).keys())
        try:
            keys |= set(getattr(wanderer, "_neuro_cfg", {}).keys())
        except Exception:
            pass
        try:
            kb = st.get("key_bank")
            if kb:
                keys |= set(kb)
        except Exception:
            pass
        # Seed known useful numeric keys into the search space without excluding any others
        keys.add("lr_override")
        keys = [k for k in set(keys) if isinstance(k, str)]
        if keys and _r.random() < mut_rate:
            k = _r.choice(keys)
            try:
                base = float((child.get("cfg", {}).get(k) if k in (child.get("cfg") or {}) else getattr(wanderer, "_neuro_cfg", {}).get(k, 0.0)))
                scale = 1.0 + ((_r.random() * 2.0 - 1.0) * (0.5 + 0.5 * (mut_rate - 0.05)))
                child["cfg"][k] = base * scale
            except Exception:
                pass
            new_pop.append(child)
        st["pop"] = new_pop
        st["gen"] = st.get("gen", 0) + 1
        st["idx"] = 0

    # ---------------- Evolution helpers (no cloning, full rollback) ----------------
    def _snapshot_graph(self, brain: "Brain") -> Dict[str, Any]:
        try:
            neurons = set(getattr(brain, "neurons", {}).keys())
        except Exception:
            neurons = set()
        try:
            syns = list(getattr(brain, "synapses", []) or [])
            syn_ids = set(id(s) for s in syns)
            syn_weights = {id(s): float(getattr(s, "weight", 1.0)) for s in syns}
        except Exception:
            syn_ids = set()
            syn_weights = {}
        return {"neurons": neurons, "syn_ids": syn_ids, "syn_w": syn_weights}

    def _restore_graph(self, brain: "Brain", snap: Dict[str, Any]) -> None:
        try:
            # Remove newly added synapses
            current_syns = list(getattr(brain, "synapses", []) or [])
            before_ids = snap.get("syn_ids", set())
            for s in current_syns:
                if id(s) not in before_ids:
                    try:
                        brain.remove_synapse(s)
                    except Exception:
                        pass
            # Remove newly added neurons
            before_neurons = snap.get("neurons", set())
            for pos, n in list(getattr(brain, "neurons", {}).items()):
                if pos not in before_neurons:
                    try:
                        brain.remove_neuron(n)
                    except Exception:
                        pass
            # Restore original synapse weights
            for s in list(getattr(brain, "synapses", []) or []):
                sid = id(s)
                if sid in snap.get("syn_w", {}):
                    try:
                        s.weight = float(snap["syn_w"][sid])
                    except Exception:
                        pass
        except Exception:
            pass

    def _apply_genome_temp(self, wanderer: "Wanderer", genome: Dict[str, Any]) -> Dict[str, Any]:
        # Apply stacks and paradigms temporarily; merge cfg; return a handle for rollback
        handle = push_temporary_plugins(wanderer, wanderer_types=genome.get("w"), neuro_types=genome.get("n"))
        prev_cfg = dict(getattr(wanderer, "_neuro_cfg", {}) or {})
        toggled = []
        try:
            for nm in genome.get("p", []) or []:
                wanderer.brain.enable_paradigm(nm, enabled=True)
                toggled.append(nm)
        except Exception:
            pass
        try:
            for k, v in (genome.get("cfg") or {}).items():
                wanderer._neuro_cfg[k] = v
        except Exception:
            pass
        return {"handle": handle, "cfg_prev": prev_cfg, "paradigms": toggled}

    def _rollback_temp(self, wanderer: "Wanderer", temp: Dict[str, Any]) -> None:
        try:
            if temp.get("handle") is not None:
                pop_temporary_plugins(wanderer, temp["handle"])
        except Exception:
            pass
        try:
            for nm in temp.get("paradigms", []) or []:
                wanderer.brain.enable_paradigm(nm, enabled=False)
        except Exception:
            pass
        try:
            wanderer._neuro_cfg = temp.get("cfg_prev", {})
        except Exception:
            pass

    def _short_eval(self, wanderer: "Wanderer", start: "Neuron", max_steps: int) -> Dict[str, float]:
        # Run a short walk with lr=0 to measure loss and speed proxy; rollback handled by caller
        stats = {"loss": float("inf"), "speed": 0.0}
        try:
            res = wanderer.walk(max_steps=max(1, int(max_steps)), start=start, lr=0.0)
            # Compute speed proxy from res.step_metrics
            dts = res.get("step_metrics", []) or []
            deltas = [float(m.get("delta", 0.0)) for m in dts]
            speed = max(0.0, - (sum(deltas) / max(1, len(deltas)))) if deltas else 0.0
            stats = {"loss": float(res.get("loss", 0.0)), "speed": float(speed)}
        except Exception:
            pass
        return stats

    def _mutate(self, genome: Dict[str, Any], wanderer: "Wanderer", rate: float) -> Dict[str, Any]:
        import random as _r
        g = {"w": list(genome.get("w", [])), "n": list(genome.get("n", [])), "p": list(genome.get("p", [])), "cfg": dict(genome.get("cfg", {}))}
        wnames = list(_WANDERER_TYPES.keys())
        nnames = list(_NEURO_TYPES.keys())
        pnames = list(_PARADIGM_TYPES.keys()) if '_PARADIGM_TYPES' in globals() else []
        # With rate, add/remove one plugin from each family (no specific selection bias)
        if wnames and _r.random() < rate:
            nm = _r.choice(wnames)
            if nm in g["w"] and _r.random() < 0.5:
                g["w"].remove(nm)
            else:
                if nm not in g["w"]:
                    g["w"].append(nm)
        if nnames and _r.random() < rate:
            nm = _r.choice(nnames)
            if nm in g["n"] and _r.random() < 0.5:
                g["n"].remove(nm)
            else:
                if nm not in g["n"]:
                    g["n"].append(nm)
        if pnames and _r.random() < rate:
            nm = _r.choice(pnames)
            if nm in g["p"] and _r.random() < 0.5:
                g["p"].remove(nm)
            else:
                if nm not in g["p"]:
                    g["p"].append(nm)
        # Numeric param tweak: pick an observed key or create a neutral one
        keys = set(g["cfg"].keys())
        try:
            keys |= set((getattr(wanderer, "_neuro_cfg", {}) or {}).keys())
        except Exception:
            pass
        if keys and _r.random() < rate:
            k = _r.choice(list(keys))
            try:
                base = float(g["cfg"].get(k, getattr(wanderer, "_neuro_cfg", {}).get(k, 0.0)))
                scale = 1.0 + ((_r.random() * 2.0 - 1.0) * 0.5)
                g["cfg"][k] = base * scale
            except Exception:
                pass
        return g

    def _apply_best_persistently(self, wanderer: "Wanderer") -> None:
        st = wanderer._plugin_state.get("hyper", {})
        best = st.get("best")
        if not best:
            return
        try:
            # Replace stacks
            wanderer._wplugins = []
            for nm in best.get("w", []) or []:
                plug = _WANDERER_TYPES.get(str(nm))
                if plug is not None:
                    wanderer._wplugins.append(plug)
            wanderer._neuro_plugins = []
            for nm in best.get("n", []) or []:
                nplug = _NEURO_TYPES.get(str(nm))
                if nplug is not None:
                    wanderer._neuro_plugins.append(nplug)
            # Enable paradigms
            for nm in best.get("p", []) or []:
                try:
                    wanderer.brain.enable_paradigm(nm, enabled=True)
                except Exception:
                    pass
            # Merge cfg
            for k, v in (best.get("cfg") or {}).items():
                wanderer._neuro_cfg[k] = v
        except Exception:
            pass

    def _run_evolution(self, wanderer: "Wanderer", start: "Neuron", steps: int) -> None:
        cfg = getattr(wanderer, "_neuro_cfg", {}) or {}
        eval_steps = int(cfg.get("hyper_eval_steps", 3))
        mut_rate = float(cfg.get("hyper_mut", 0.3))
        st = wanderer._plugin_state.get("hyper", {})
        # Initialize best genome from current stacks and cfg
        base = {"w": [], "n": [], "p": [], "cfg": dict(cfg)}
        # Evaluate base
        base_snap = self._snapshot_graph(wanderer.brain)
        temp = self._apply_genome_temp(wanderer, base)
        base_stats = self._short_eval(wanderer, start, eval_steps)
        self._rollback_temp(wanderer, temp)
        self._restore_graph(wanderer.brain, base_snap)
        best = base
        best_loss = float(base_stats.get("loss", float("inf")))
        best_speed = float(base_stats.get("speed", 0.0))
        for _ in range(max(1, int(steps))):
            cand = self._mutate(best, wanderer, rate=mut_rate)
            snap = self._snapshot_graph(wanderer.brain)
            temp = self._apply_genome_temp(wanderer, cand)
            stats = self._short_eval(wanderer, start, eval_steps)
            self._rollback_temp(wanderer, temp)
            self._restore_graph(wanderer.brain, snap)
            cand_loss = float(stats.get("loss", float("inf")))
            cand_speed = float(stats.get("speed", 0.0))
            # Accept only if BOTH metrics improve (strict dominance) per user objective
            if cand_loss < best_loss and cand_speed > best_speed:
                best = cand
                best_loss = cand_loss
                best_speed = cand_speed
        # Persist best in plugin state
        try:
            st["best"] = best
            st["best_loss"] = best_loss
            st["best_speed"] = best_speed
        except Exception:
            pass

__all__ = ["HyperEvolutionPlugin"]
