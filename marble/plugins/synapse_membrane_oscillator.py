from __future__ import annotations

"""Membrane oscillator synapse plugin.

Simulates an oscillatory membrane between neurons whose frequency and damping
are learnable, seeking emergent resonance patterns."""

from typing import Any, List
import math

from ..graph import Synapse
from ..wanderer import expose_learnable_params
from ..reporter import report


class MembraneOscillatorSynapsePlugin:
    """Oscillate transmissions via a fictive membrane."""

    def __init__(self) -> None:
        self._state = {}

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer,
        *,
        osc_freq: float = 1.0,
        osc_damp: float = 0.1,
    ):
        return (osc_freq, osc_damp)

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
        freq, damp = 1.0, 0.1
        if wanderer is not None:
            freq, damp = self._params(wanderer)
        freq_f = float(freq.detach().to("cpu").item()) if hasattr(freq, "detach") else float(freq)
        damp_f = float(damp.detach().to("cpu").item()) if hasattr(damp, "detach") else float(damp)

        phase, amp = self._state.get(id(syn), (0.0, 0.0))
        vals = self._to_list(value)
        out_vals = []
        for v in vals:
            amp = amp * (1 - damp_f) + v
            phase += freq_f
            out_vals.append(amp * math.sin(phase))
        self._state[id(syn)] = (phase, amp)
        out = out_vals if len(out_vals) != 1 else out_vals[0]

        try:
            report(
                "synapse",
                "membrane_oscillator",
                {"freq": freq_f, "damp": damp_f},
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


__all__ = ["MembraneOscillatorSynapsePlugin"]
