from __future__ import annotations

"""Entropy lens synapse plugin.

Filters transmissions through an entropy-based lens that sharpens or blurs
signals according to a learnable focus parameter."""

from typing import Any, List
import math

from ..graph import Synapse
from ..wanderer import expose_learnable_params
from ..reporter import report


class EntropyLensSynapsePlugin:
    """Adjust transmission magnitude via entropy lensing."""

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer,
        *,
        lens_focus: float = 0.5,
    ):
        return (lens_focus,)

    def _to_list(self, value: Any) -> List[float]:
        if hasattr(value, "detach") and hasattr(value, "tolist"):
            return [float(v) for v in value.detach().to("cpu").view(-1).tolist()]
        if isinstance(value, (list, tuple)):
            return [float(v) for v in value]
        return [float(value)]

    def transmit(self, syn: "Synapse", value: Any, *, direction: str = "forward") -> Any:
        wanderer = getattr(getattr(syn.source, "_plugin_state", {}), "get", lambda *_: None)("wanderer")
        if wanderer is None:
            wanderer = getattr(getattr(syn.target, "_plugin_state", {}), "get", lambda *_: None)("wanderer")
        focus = 0.5
        if wanderer is not None:
            (focus,) = self._params(wanderer)
        focus_f = float(focus.detach().to("cpu").item()) if hasattr(focus, "detach") else float(focus)

        vals = self._to_list(value)
        out_vals = []
        for v in vals:
            mag = abs(v)
            entropy = -mag * math.log(mag + 1e-6)
            out_vals.append(v + entropy * focus_f)
        out = out_vals if len(out_vals) != 1 else out_vals[0]

        try:
            report("synapse", "entropy_lens", {"focus": focus_f}, "plugins")
        except Exception:
            pass

        orig = syn.type_name
        syn.type_name = None
        try:
            return Synapse.transmit(syn, out, direction=direction)
        finally:
            syn.type_name = orig


__all__ = ["EntropyLensSynapsePlugin"]
