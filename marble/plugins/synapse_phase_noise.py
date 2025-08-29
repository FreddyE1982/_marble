from __future__ import annotations

"""Phase noise synapse plugin.

Superimposes a sinusoidal perturbation whose phase advances each
transmission, akin to a drifting interference pattern. Both frequency and
amplitude of the noise are learnable.
"""

import math
from typing import Any, List

from ..graph import Synapse
from ..wanderer import expose_learnable_params
from ..reporter import report


class PhaseNoiseSynapsePlugin:
    """Adds progressive phase noise to transmissions."""

    def __init__(self) -> None:
        self._phase = {}

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer,
        *,
        noise_freq: float = 1.0,
        noise_amp: float = 0.1,
    ) -> Any:
        return (noise_freq, noise_amp)

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
        freq, amp = 1.0, 0.1
        if wanderer is not None:
            freq, amp = self._params(wanderer)
        freq_f = float(freq.detach().to("cpu").item()) if hasattr(freq, "detach") else float(freq)
        amp_f = float(amp.detach().to("cpu").item()) if hasattr(amp, "detach") else float(amp)

        ph = self._phase.get(id(syn), 0.0)
        vals = self._to_list(value)
        out_vals: List[float] = []
        for v in vals:
            ph += freq_f
            out_vals.append(v + math.sin(ph) * amp_f)
        self._phase[id(syn)] = ph
        out = out_vals if len(out_vals) != 1 else out_vals[0]

        try:
            report(
                "synapse",
                "phase_noise",
                {"freq": freq_f, "amp": amp_f},
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


__all__ = ["PhaseNoiseSynapsePlugin"]

