from __future__ import annotations

"""Dropout-style synapse plugin.

Randomly drops the transmitted value according to a learnable probability.
Uses :func:`expose_learnable_params` so optimisation frameworks can tune the
drop rate during training.
"""

from typing import Any, List
import random

from ..graph import register_synapse_type, Synapse
from ..wanderer import expose_learnable_params
from ..reporter import report


class DropoutSynapsePlugin:
    """Zeroes transmissions with a learnable probability."""

    @staticmethod
    @expose_learnable_params
    def _params(wanderer, *, dropout_p: float = 0.1) -> Any:
        return (dropout_p,)

    def _to_list(self, value: Any) -> List[float]:
        if hasattr(value, "detach") and hasattr(value, "tolist"):
            return [float(v) for v in value.detach().to("cpu").view(-1).tolist()]
        if isinstance(value, (list, tuple)):
            return [float(v) for v in value]
        return [float(value)]

    def transmit(self, syn: "Synapse", value: Any, *, direction: str = "forward") -> Any:
        # Obtain wanderer from source/target plugin state
        wanderer = getattr(getattr(syn.source, "_plugin_state", {}), "get", lambda *_: None)("wanderer")
        if wanderer is None:
            wanderer = getattr(getattr(syn.target, "_plugin_state", {}), "get", lambda *_: None)("wanderer")
        drop = 0.0
        if wanderer is not None:
            (drop,) = self._params(wanderer)

        torch = getattr(syn, "_torch", None)
        dev = getattr(syn, "_device", "cpu")
        if torch is not None:
            vt = value
            if not (hasattr(vt, "detach") and hasattr(vt, "to")):
                vt = torch.tensor(self._to_list(value), dtype=torch.float32, device=dev)
            mask = (torch.rand_like(vt) > drop).float()
            out = vt * mask
        else:
            vals = self._to_list(value)
            out = [0.0 if random.random() < float(drop) else v for v in vals]
            if len(out) == 1:
                out = out[0]

        try:
            report(
                "synapse",
                "dropout_transmit",
                {"p": float(drop.detach().to("cpu").item()) if hasattr(drop, "detach") else float(drop)},
                "plugins",
            )
        except Exception:
            pass

        orig = syn.type_name
        syn.type_name = None
        try:
            return Synapse.transmit(syn, out, direction=direction)
        finally:
            syn.type_name = orig


try:
    register_synapse_type("dropout", DropoutSynapsePlugin())
except Exception:
    pass

__all__ = ["DropoutSynapsePlugin"]

