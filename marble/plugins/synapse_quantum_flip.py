from __future__ import annotations

"""Quantum flip synapse plugin.

Applies a sign flip to transmissions with a learnable probability. The effect
resembles a measurement collapsing a quantum state, introducing stochastic
parity changes along connections.
"""

import random
from typing import Any, List

from ..graph import Synapse
from ..wanderer import expose_learnable_params
from ..reporter import report


class QuantumFlipSynapsePlugin:
    """Randomly flips the sign of transmitted values."""

    @staticmethod
    @expose_learnable_params
    def _params(wanderer, *, flip_prob: float = 0.5) -> Any:
        return (flip_prob,)

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
        prob = 0.5
        if wanderer is not None:
            (prob,) = self._params(wanderer)
        prob_f = float(prob.detach().to("cpu").item()) if hasattr(prob, "detach") else float(prob)

        torch = getattr(syn, "_torch", None)
        dev = getattr(syn, "_device", "cpu")
        if torch is not None:
            vt = value
            if not (hasattr(vt, "detach") and hasattr(vt, "to")):
                vt = torch.tensor(self._to_list(value), dtype=torch.float32, device=dev)
            mask = torch.where(torch.rand_like(vt) < prob_f, -1.0, 1.0)
            out = vt * mask
        else:
            vals = self._to_list(value)
            out = [(-v if random.random() < prob_f else v) for v in vals]
            if len(out) == 1:
                out = out[0]

        try:
            report(
                "synapse",
                "quantum_flip",
                {"p": prob_f},
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


__all__ = ["QuantumFlipSynapsePlugin"]

