from __future__ import annotations

"""Spike gating synapse plugin.

Applies a smooth threshold via a logistic gate. Two learnable parameters control
the threshold and sharpness of the gate.
"""

from typing import Any, List
import math

from ..graph import register_synapse_type, Synapse
from ..wanderer import expose_learnable_params
from ..reporter import report


class SpikeGateSynapsePlugin:
    """Logistic gating with learnable threshold and sharpness."""

    @staticmethod
    @expose_learnable_params
    def _params(wanderer, *, gate_thresh: float = 0.5, gate_sharp: float = 10.0) -> Any:
        return (gate_thresh, gate_sharp)

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
        thresh, sharp = 0.5, 10.0
        if wanderer is not None:
            thresh, sharp = self._params(wanderer)

        vals = self._to_list(value)
        torch = getattr(syn, "_torch", None)
        dev = getattr(syn, "_device", "cpu")
        if torch is not None:
            vt = value
            if not (hasattr(vt, "detach") and hasattr(vt, "to")):
                vt = torch.tensor(vals, dtype=torch.float32, device=dev)
            gate = torch.sigmoid((vt - thresh) * sharp)
            out = vt * gate
        else:
            gate = [1.0 / (1.0 + math.exp(-((v - float(thresh)) * float(sharp)))) for v in vals]
            out_vals = [v * g for v, g in zip(vals, gate)]
            out = out_vals if len(out_vals) != 1 else out_vals[0]

        try:
            report(
                "synapse",
                "spike_gate",
                {
                    "thresh": float(thresh.detach().to("cpu").item()) if hasattr(thresh, "detach") else float(thresh),
                    "sharp": float(sharp.detach().to("cpu").item()) if hasattr(sharp, "detach") else float(sharp),
                },
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
    register_synapse_type("spike_gate", SpikeGateSynapsePlugin())
except Exception:
    pass

__all__ = ["SpikeGateSynapsePlugin"]

