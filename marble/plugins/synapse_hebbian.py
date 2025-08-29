from __future__ import annotations

"""Hebbian synapse plugin.

Applies a simple Hebbian update to the synapse weight based on the mean of the
pre- and post-synaptic activations. Two learnable parameters control the update
rate and weight decay.
"""

from typing import Any, List

from ..graph import Synapse
from ..wanderer import expose_learnable_params
from ..reporter import report


class HebbianSynapsePlugin:
    """Online Hebbian rule with learnable rate and decay."""

    @staticmethod
    @expose_learnable_params
    def _params(wanderer, *, hebb_rate: float = 0.01, hebb_decay: float = 0.0) -> Any:
        return (hebb_rate, hebb_decay)

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
        rate, decay = 0.0, 0.0
        if wanderer is not None:
            rate, decay = self._params(wanderer)

        pre_vals = self._to_list(value)
        orig = syn.type_name
        syn.type_name = None
        try:
            out_neuron = Synapse.transmit(syn, value, direction=direction)
        finally:
            syn.type_name = orig

        try:
            post_vals = self._to_list(getattr(out_neuron, "tensor", 0.0))
            pre_mean = sum(pre_vals) / max(1, len(pre_vals))
            post_mean = sum(post_vals) / max(1, len(post_vals))
            update = pre_mean * post_mean * rate
            update_f = (
                float(update.detach().to("cpu").item())
                if hasattr(update, "detach")
                else float(update)
            )
            decay_f = (
                float(decay.detach().to("cpu").item())
                if hasattr(decay, "detach")
                else float(decay)
            )
            syn.weight += update_f
            syn.weight *= (1.0 - decay_f)
            report(
                "synapse",
                "hebbian_update",
                {
                    "rate": float(rate.detach().to("cpu").item()) if hasattr(rate, "detach") else float(rate),
                    "decay": decay_f,
                },
                "plugins",
            )
        except Exception:
            pass

        return out_neuron


__all__ = ["HebbianSynapsePlugin"]

