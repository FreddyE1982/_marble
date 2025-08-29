from __future__ import annotations

from typing import Any, List

# Explicit registration name so auto-discovery exposes this plugin under
# "alternatepathscreator", matching the name used in tests and code.
PLUGIN_NAME = "alternatepathscreator"

from ..reporter import report


class AlternatePathsCreatorPlugin:
    """At the end of each walk, create a new alternate path of random length and connect it to a random visited neuron.

    Config keys read from `wanderer._neuro_cfg`:
    - `altpaths_min_len` (default 2)
    - `altpaths_max_len` (default 4)
    """

    def on_walk_end(self, wanderer: "Wanderer", stats: Dict[str, Any]) -> None:
        import random as _rand
        brain = wanderer.brain
        visited = getattr(wanderer, "_visited", []) or []
        if not visited:
            return
        cfg = getattr(wanderer, "_neuro_cfg", {}) or {}
        min_len = int(cfg.get("altpaths_min_len", 2))
        max_len = int(cfg.get("altpaths_max_len", 4))
        if max_len < min_len:
            max_len = min_len
        # Per-walk limit on number of new paths
        max_paths = int(cfg.get("altpaths_max_paths_per_walk", 1))
        created = int(getattr(wanderer, "_plugin_state", {}).get("altpaths_created", 0))
        if created >= max_paths:
            return
        length = _rand.randint(min_len, max_len)
        # Pick an anchor neuron from the visited path
        anchor = _rand.choice(visited)
        # Build a chain of new neurons
        new_nodes = []
        try:
            for _ in range(length):
                # pick first free index
                idx = None
                for cand in brain.available_indices():
                    try:
                        if brain.get_neuron(cand) is None:
                            idx = cand
                            break
                    except Exception:
                        continue
                if idx is None:
                    # try neighbor of last node (or anchor)
                    base = new_nodes[-1] if new_nodes else anchor
                    bpos = getattr(base, "position", None)
                    if isinstance(bpos, tuple):
                        for dx, dy in [(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,-1)]:
                            cand = (bpos[0]+dx, bpos[1]+dy) if len(bpos)>=2 else (bpos[0]+dx,)
                            try:
                                if brain.is_inside(cand) and brain.get_neuron(cand) is None:
                                    idx = cand
                                    break
                            except Exception:
                                continue
                if idx is None:
                    break
                n = brain.add_neuron(idx, tensor=0.0)
                new_nodes.append(n)
        except Exception:
            new_nodes = new_nodes
        if not new_nodes:
            return
        # Connect chain
        try:
            prev = anchor
            for node in new_nodes:
                brain.connect(getattr(prev, "position"), getattr(node, "position"), direction="uni")
                prev = node
            report("training", "altpaths_create", {"anchor": getattr(anchor, "position", None), "len": len(new_nodes)}, "events")
            # Track created count this walk
            if not getattr(wanderer, "_plugin_state", None):
                wanderer._plugin_state = {}
            wanderer._plugin_state["altpaths_created"] = created + 1
        except Exception:
            pass

__all__ = ["AlternatePathsCreatorPlugin"]
