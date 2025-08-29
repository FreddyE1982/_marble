from __future__ import annotations

"""Quantum tunnel style neuron plugin.

This highly experimental activation mimics quantum tunneling by
exponentially damping values based on a learnable barrier height.
"""

from typing import Any, Tuple
import math

from ..wanderer import expose_learnable_params
from ..reporter import report


class QuantumTunnelNeuronPlugin:
    """Apply exponential damping with a learnable barrier."""

    @staticmethod
    @expose_learnable_params
    def _params(wanderer: "Wanderer", *, qt_barrier: float = 1.0) -> Tuple[Any]:
        return (qt_barrier,)

    def forward(self, neuron: "Neuron", input_value=None):
        x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)

        barrier = 1.0
        wanderer = neuron._plugin_state.get("wanderer")
        if wanderer is not None:
            try:
                (barrier,) = self._params(wanderer)
            except Exception:
                pass

        torch = getattr(neuron, "_torch", None)
        if torch is not None and neuron._is_torch_tensor(x):
            y = x * torch.exp(-barrier * torch.abs(x))
            try:
                report(
                    "neuron",
                    "quantum_tunnel_forward",
                    {"barrier": float(barrier.detach().to("cpu").item()) if hasattr(barrier, "detach") else float(barrier)},
                    "plugins",
                )
            except Exception:
                pass
            return y

        x_list = x if isinstance(x, list) else [float(x)]
        barrier_f = (
            float(barrier.detach().to("cpu").item()) if hasattr(barrier, "detach") else float(barrier)
        )
        out = [v * math.exp(-barrier_f * abs(v)) for v in x_list]
        try:
            report("neuron", "quantum_tunnel_forward", {"barrier": barrier_f}, "plugins")
        except Exception:
            pass
        return out if len(out) != 1 else out[0]


__all__ = ["QuantumTunnelNeuronPlugin"]

