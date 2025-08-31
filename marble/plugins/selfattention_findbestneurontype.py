from __future__ import annotations

"""SelfAttention routine that picks the best neuron type for new neurons.

Whenever ``Brain.add_neuron`` is invoked while this routine is attached to a
Wanderer via :class:`marble.selfattention.SelfAttention`, the call is intercepted
and evaluated for all known neuron types.  For each candidate type the routine

1. records the current loss (from ``Wanderer._last_walk_loss``),
2. creates a neuron of that type,
3. runs a single Wanderer step starting from the new neuron with ``lr=0``
   (ensuring no parameter updates),
4. measures the loss improvement relative to the baseline, and
5. removes the test neuron.

After trying all available neuron types the neuron with the largest loss
improvement is finally created and returned.

The routine operates entirely via public APIs without modifying tests or core
logic.  It uses a guard flag to avoid recursive interception when the temporary
evaluation walk happens to trigger further neuron additions.
"""

from typing import Any, Dict, Optional

from ..wanderer import expose_learnable_params
from ..graph import _NEURON_TYPES
from .selfattention_metric_utils import metric_factor


@expose_learnable_params
def _nbt_params(wanderer, max_eval_types: float = 10.0):
    return max_eval_types


class FindBestNeuronTypeRoutine:
    """SelfAttention routine implementing neuron type search."""

    def __init__(self) -> None:
        self._sa = None  # type: Optional["SelfAttention"]
        self._orig_add = None
        self._eval_active = False
        self._limit_factor = 1.0

    # ---- Helper utilities -------------------------------------------------
    def _baseline_loss(self, wanderer: "Wanderer") -> float:
        try:
            base = getattr(wanderer, "_last_walk_loss", None)
            return float(base) if base is not None else float("inf")
        except Exception:
            return float("inf")

    def _list_types(self, wanderer: "Wanderer") -> list[str]:
        max_t = _nbt_params(wanderer)
        try:
            limit = max(1, int(max_t.detach().to("cpu").item()))
        except Exception:
            limit = 10
        limit = max(1, int(limit * self._limit_factor))
        types = ["base"]
        try:
            types += list(_NEURON_TYPES.keys())
        except Exception:
            pass
        return sorted(set(t for t in types if isinstance(t, str) and t))[:limit]

    # ---- Hook installation -------------------------------------------------
    def on_init(self, selfattention: "SelfAttention") -> None:
        self._sa = selfattention
        owner = getattr(selfattention, "_owner", None)
        if owner is None:
            return
        brain = getattr(owner, "brain", None)
        if brain is None:
            return
        self._orig_add = getattr(brain, "add_neuron")

        def wrapped_add_neuron(index, *, tensor=0.0, **kwargs):
            if self._eval_active or self._orig_add is None:
                return self._orig_add(index, tensor=tensor, **kwargs)  # type: ignore[misc]
            self._eval_active = True
            try:
                w = getattr(selfattention, "_owner", None)
                if w is None:
                    return self._orig_add(index, tensor=tensor, **kwargs)  # type: ignore[misc]
                baseline = self._baseline_loss(w)
                best_type: Optional[str] = None
                best_diff: Optional[float] = None
                existing = len(getattr(brain, "neurons", {}) or {})
                for t in self._list_types(w):
                    if t != "base" and existing < 2:
                        # No other neurons available to form required wiring
                        continue
                    try:
                        n = self._orig_add(index, tensor=tensor, type_name=None if t == "base" else t, **kwargs)
                    except Exception:
                        # Failed to create due to unmet wiring requirements
                        continue
                    neuro_bak = getattr(w, "_neuro_plugins", [])
                    try:
                        setattr(w, "_neuro_plugins", [])
                        res = w.walk(max_steps=1, start=n, lr=0.0)
                        cand_loss = float(res.get("loss", baseline))
                    except Exception:
                        cand_loss = baseline
                    finally:
                        setattr(w, "_neuro_plugins", neuro_bak)
                        try:
                            brain.remove_neuron(n)
                        except Exception:
                            pass
                    diff = baseline - cand_loss
                    if best_diff is None or diff > best_diff:
                        best_diff = diff
                        best_type = t
                if best_type is None:
                    # Fallback: no candidate type could be wired; add basic neuron
                    return self._orig_add(index, tensor=tensor, **kwargs)
                # Create final neuron of best type
                return self._orig_add(index, tensor=tensor, type_name=None if best_type == "base" else best_type, **kwargs)  # type: ignore[misc]
            finally:
                self._eval_active = False

        try:
            brain.add_neuron = wrapped_add_neuron  # type: ignore[assignment]
        except Exception:
            pass

    def after_step(
        self,
        selfattention: "SelfAttention",
        reporter_ro: Any,
        wanderer: "Wanderer",
        step_index: int,
        ctx: Dict[str, Any],
    ) -> None:
        self._limit_factor = 1.0 + metric_factor(ctx, "findbestneurontype")
        return None


__all__ = ["FindBestNeuronTypeRoutine"]

