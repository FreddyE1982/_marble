from __future__ import annotations

"""Echo mix neuron plugin.

Blends the current input with a decaying echo of the previous output,
creating a simple memory effect controlled by a learnable ``memory``
coefficient.
"""

from typing import Any, Tuple

from ..wanderer import expose_learnable_params
from ..reporter import report


class EchoMixNeuronPlugin:
    """Mix current input with a stored echo."""

    @staticmethod
    @expose_learnable_params
    def _params(wanderer: "Wanderer", *, echo_memory: float = 0.5) -> Tuple[Any]:
        return (echo_memory,)

    def forward(self, neuron: "Neuron", input_value=None):
        x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)

        mem = 0.5
        wanderer = neuron._plugin_state.get("wanderer")
        if wanderer is not None:
            try:
                (mem,) = self._params(wanderer)
            except Exception:
                pass

        prev = neuron._plugin_state.get("echo_prev")

        torch = getattr(neuron, "_torch", None)
        if torch is not None and neuron._is_torch_tensor(x):
            prev_t = (
                prev
                if prev is not None and neuron._is_torch_tensor(prev)
                else torch.zeros_like(x)
            )
            y = (1 - mem) * x + mem * prev_t
            neuron._plugin_state["echo_prev"] = y.detach()
            try:
                report(
                    "neuron",
                    "echo_mix_forward",
                    {"memory": float(mem.detach().to("cpu").item()) if hasattr(mem, "detach") else float(mem)},
                    "plugins",
                )
            except Exception:
                pass
            return y

        x_list = x if isinstance(x, list) else [float(x)]
        prev_list = (
            prev if isinstance(prev, list) else [float(prev)] if prev is not None else [0.0]
        )
        mem_f = float(mem.detach().to("cpu").item()) if hasattr(mem, "detach") else float(mem)
        out = [
            (1 - mem_f) * xv + mem_f * pv
            for xv, pv in zip(x_list, prev_list + [0.0] * (len(x_list) - len(prev_list)))
        ]
        neuron._plugin_state["echo_prev"] = out if len(out) != 1 else out[0]
        try:
            report("neuron", "echo_mix_forward", {"memory": mem_f}, "plugins")
        except Exception:
            pass
        return out if len(out) != 1 else out[0]


__all__ = ["EchoMixNeuronPlugin"]

