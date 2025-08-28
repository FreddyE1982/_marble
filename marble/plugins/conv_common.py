from __future__ import annotations

"""Shared helpers for n-dimensional convolution-style neuron plugins."""

from typing import Optional


class _ConvNDCommon:
    """Utility mixin with tensor/list helpers used by conv/pool plugins."""

    def _key_src(self, syn: "Synapse"):
        src = syn.source
        pos = getattr(src, "position", None)
        if isinstance(pos, tuple):
            return (0, tuple(pos))
        return (1, id(src))

    def _to_list1d(self, x) -> list:
        try:
            if hasattr(x, "detach") and hasattr(x, "tolist"):
                lst = x.detach().to("cpu").view(-1).tolist()
            elif isinstance(x, (list, tuple)):
                def flat(z):
                    for it in z:
                        if isinstance(it, (list, tuple)):
                            yield from flat(it)
                        else:
                            yield it
                lst = list(flat(list(x)))
            else:
                lst = [x]
        except Exception:
            lst = []
        out = []
        for v in lst:
            try:
                out.append(float(v))
            except Exception:
                out.append(0.0)
        return out

    def _first_scalar(self, x, *, default: float = 0.0, min_val: Optional[float] = None) -> float:
        vals = self._to_list1d(x)
        v = float(vals[0]) if vals else float(default)
        if min_val is not None and v < min_val:
            v = float(min_val)
        return v


__all__ = ["_ConvNDCommon"]

