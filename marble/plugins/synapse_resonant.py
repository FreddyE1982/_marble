from __future__ import annotations

"""Resonant synapse plugin.

Models a damped harmonic oscillator that filters the incoming signal. The
oscillation frequency and damping factor are learnable, enabling the network to
specialise transmission dynamics.
"""

from typing import Any, List

from ..graph import register_synapse_type, Synapse
from ..wanderer import expose_learnable_params
from ..reporter import report


class ResonantSynapsePlugin:
    """Damped oscillator filter with learnable frequency and damping."""

    def __init__(self) -> None:
        self._state = {}

    @staticmethod
    @expose_learnable_params
    def _params(wanderer, *, res_freq: float = 1.0, res_damp: float = 0.1) -> Any:
        return (res_freq, res_damp)

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

        vals = self._to_list(value)
        st = self._state.setdefault(id(syn), {"pos": 0.0, "vel": 0.0})
        out_vals: List[float] = []
        for v in vals:
            vel = st["vel"] + freq_f * (v - st["pos"]) - damp_f * st["vel"]
            pos = st["pos"] + vel
            st["vel"], st["pos"] = vel, pos
            out_vals.append(pos)
        out = out_vals if len(out_vals) != 1 else out_vals[0]

        try:
            report(
                "synapse",
                "resonant_step",
                {
                    "freq": freq_f,
                    "damp": damp_f,
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
    register_synapse_type("resonant", ResonantSynapsePlugin())
except Exception:
    pass

__all__ = ["ResonantSynapsePlugin"]

