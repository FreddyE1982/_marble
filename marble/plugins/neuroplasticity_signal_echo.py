"""SignalEchoPlugin feeds back a fraction of the last output via a learnable strength."""

from __future__ import annotations

from typing import Any

from ..wanderer import expose_learnable_params
from ..reporter import report


@expose_learnable_params
def _echo_param(wanderer, echo_strength: float = 0.1):
    return echo_strength


class SignalEchoPlugin:
    """Add a decaying echo of previous output to the current synapse weight."""

    def __init__(self) -> None:
        self._last = 0.0

    def on_step(self, wanderer: "Wanderer", current: "Neuron", syn: "Synapse", direction: str, step_index: int, out_value: Any) -> None:
        es_t = _echo_param(wanderer)
        try:
            es = float(es_t.detach().to("cpu").item())
        except Exception:
            es = 0.0
        try:
            syn.weight = float(getattr(syn, "weight", 0.0)) + self._last * es
            report("neuroplasticity", "signal_echo", {"step": int(step_index), "echo": es}, "plugins")
        except Exception:
            pass
        try:
            val = out_value
            if hasattr(val, "detach"):
                val = float(val.detach().to("cpu").item())
            else:
                val = float(val)
            self._last = val
        except Exception:
            self._last = 0.0
        return None


__all__ = ["SignalEchoPlugin"]

