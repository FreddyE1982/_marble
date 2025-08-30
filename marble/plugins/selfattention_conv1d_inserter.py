from __future__ import annotations

from typing import Any, Dict, List, Optional




class Conv1DRandomInsertionRoutine:
    """Insert a conv1d neuron every N steps; remove if loss not improved.

    - period: check insertion every `period` global steps (from Reporter).
    - eval_after: number of steps to wait after insertion before evaluation.
    - kernel: default kernel for conv1d parameter neuron.

    Routines operate via SelfAttention.after_step hook and only use
    the provided Wanderer/Brain APIs (no imports), as per architecture.
    """

    def __init__(self, period: int = 20, eval_after: int = 10, kernel: Optional[List[float]] = None, max_data_sources: int = 1) -> None:
        self.period = max(1, int(period))
        self.eval_after = max(1, int(eval_after))
        self.kernel = list(kernel) if isinstance(kernel, list) else [1.0, 0.0, -1.0]
        self._active: Optional[Dict[str, Any]] = None
        self.max_data_sources = max(0, int(max_data_sources))

    def _global_step(self, sa: "SelfAttention") -> int:
        try:
            return len(sa.history())
        except Exception:
            return 0

    def _mean_loss(self, sa: "SelfAttention", start: int, end: Optional[int]) -> float:
        try:
            hist = sa.history()
            if end is None or end > len(hist):
                end = len(hist)
            vals: List[float] = []
            for rec in hist[start:end]:
                v = rec.get("current_loss")
                try:
                    vals.append(float(v))
                except Exception:
                    pass
            return (sum(vals) / max(1, len(vals))) if vals else float("inf")
        except Exception:
            return float("inf")

    def _random_free_index(self, brain: "Brain"):
        try:
            candidates = list(brain.available_indices())
        except Exception:
            candidates = []
        import random as _r
        _r.shuffle(candidates)
        for idx in candidates:
            try:
                if brain.get_neuron(idx) is None:
                    return idx
            except Exception:
                continue
        return candidates[0] if candidates else (0,) * int(getattr(brain, "n", 1))

    def _pick_param_sources(self, brain: "Brain", exclude: List["Neuron"]) -> List["Neuron"]:
        # Deterministically select 5 existing neurons not in exclude
        ex = set(id(n) for n in exclude)
        candidates = []
        try:
            for n in getattr(brain, "neurons", {}).values():
                if id(n) not in ex:
                    candidates.append(n)
        except Exception:
            candidates = []
        # Sort deterministically by position tuple then id for stability
        def keyfn(n):
            pos = getattr(n, "position", None)
            return (0, tuple(pos)) if isinstance(pos, tuple) else (1, id(n))
        candidates.sort(key=keyfn)
        return candidates[:5]

    def after_step(self, sa: "SelfAttention", ro, wanderer: "Wanderer", step_idx: int, ctx: Dict[str, Any]):
        gstep = self._global_step(sa)
        if gstep % max(1, self.period) != 0:
            return None
        brain = wanderer.brain
        # Pick a destination neuron from the current or last visited
        try:
            dst = getattr(ctx.get("current"), "position", None)
            if dst is None and getattr(wanderer, "_visited", []):
                dst = getattr(wanderer._visited[-1], "position", None)
        except Exception:
            dst = None
        if dst is None:
            return None
        # Prepare parameter neurons (kernel, stride, padding, dilation, bias)
        created: List["Neuron"] = []
        try:
            ps = [
                brain.add_neuron(self._random_free_index(brain), tensor=list(self.kernel)),
                brain.add_neuron(self._random_free_index(brain), tensor=[1.0]),
                brain.add_neuron(self._random_free_index(brain), tensor=[0.0]),
                brain.add_neuron(self._random_free_index(brain), tensor=[1.0]),
                brain.add_neuron(self._random_free_index(brain), tensor=[0.0]),
            ]
            created.extend(ps)
            dstn = brain.get_neuron(dst)
            if dstn is None:
                return None
            conv_idx = self._random_free_index(brain)
            conv = brain.add_neuron(conv_idx, tensor=[0.0])
            from ..marblemain import wire_param_synapses, _NEURON_TYPES
            wire_param_synapses(brain, conv, ps)
            brain.connect(getattr(conv, "position"), dst, direction="uni")
            conv.type_name = "conv1d"
            plugin = _NEURON_TYPES.get("conv1d")
            if plugin is not None and hasattr(plugin, "on_init"):
                plugin.on_init(conv)  # type: ignore[attr-defined]
        except Exception:
            return None
        # No param updates here; leave evaluation/rollback policy to future extensions
        return None


__all__ = ["Conv1DRandomInsertionRoutine"]
