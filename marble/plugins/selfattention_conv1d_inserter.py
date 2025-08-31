from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..wanderer import expose_learnable_params
from .selfattention_metric_utils import metric_factor


@expose_learnable_params
def _conv1d_params(
    wanderer,
    period: float = 20.0,
    eval_after: float = 10.0,
    max_data_sources: float = 1.0,
):
    return period, eval_after, max_data_sources


class Conv1DRandomInsertionRoutine:
    """Insert a conv1d neuron periodically and evaluate its effect.

    Core parameters are exposed as learnables to allow automated tuning.
    """

    def __init__(
        self,
        period: int = 20,
        eval_after: int = 10,
        kernel: Optional[List[float]] = None,
        max_data_sources: int = 1,
    ) -> None:
        self._period_def = period
        self._eval_after_def = eval_after
        self.kernel = list(kernel) if isinstance(kernel, list) else [1.0, 0.0, -1.0]
        self._active: Optional[Dict[str, Any]] = None
        self._max_data_sources_def = max_data_sources

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
        ex = set(id(n) for n in exclude)
        candidates = []
        try:
            for n in getattr(brain, "neurons", {}).values():
                if id(n) not in ex:
                    candidates.append(n)
        except Exception:
            candidates = []
        def keyfn(n):
            pos = getattr(n, "position", None)
            return (0, tuple(pos)) if isinstance(pos, tuple) else (1, id(n))
        candidates.sort(key=keyfn)
        return candidates[:5]

    def after_step(self, sa: "SelfAttention", ro, wanderer: "Wanderer", step_idx: int, ctx: Dict[str, Any]):
        p_t, e_t, m_t = _conv1d_params(
            wanderer, self._period_def, self._eval_after_def, self._max_data_sources_def
        )
        try:
            period = max(1, int(p_t.detach().to("cpu").item()))
            eval_after = max(1, int(e_t.detach().to("cpu").item()))
            self.max_data_sources = max(0, int(m_t.detach().to("cpu").item()))
        except Exception:
            period = max(1, int(self._period_def))
            eval_after = max(1, int(self._eval_after_def))
            self.max_data_sources = max(0, int(self._max_data_sources_def))
        factor = metric_factor(ctx, "conv1d_inserter")
        self.period = max(1, int(period * (1.0 + factor)))
        self.eval_after = max(1, int(eval_after * (1.0 + factor)))

        gstep = self._global_step(sa)
        if gstep % max(1, self.period) != 0:
            return None
        brain = wanderer.brain
        try:
            dst = getattr(ctx.get("current"), "position", None)
            if dst is None and getattr(wanderer, "_visited", []):
                dst = getattr(wanderer._visited[-1], "position", None)
        except Exception:
            dst = None
        if dst is None:
            return None
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
        return None


__all__ = ["Conv1DRandomInsertionRoutine"]

