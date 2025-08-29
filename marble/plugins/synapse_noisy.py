from __future__ import annotations

from typing import Any, List

from ..graph import Synapse
from ..reporter import report


class NoisySynapsePlugin:
    """Adds zero-mean Gaussian noise to transmitted values.

    Config via per-synapse state:
      - synapse._plugin_state['sigma'] (float, default 0.01)
    """

    def on_init(self, syn: "Synapse") -> None:
        st = getattr(syn, "_plugin_state", None)
        if st is None:
            syn._plugin_state = {}
            st = syn._plugin_state
        st.setdefault("sigma", 0.01)
        try:
            report("synapse", "noisy_init", {"sigma": float(st["sigma"])}, "plugins")
        except Exception:
            pass

    def _to_list(self, syn: "Synapse", value: Any) -> List[float]:
        try:
            if hasattr(value, "detach") and hasattr(value, "tolist"):
                return [float(v) for v in value.detach().to("cpu").view(-1).tolist()]
            if isinstance(value, (list, tuple)):
                return [float(v) for v in value]
            return [float(value)]
        except Exception:
            return []

    def transmit(self, syn: "Synapse", value: Any, *, direction: str = "forward") -> None:
        sigma = float(getattr(getattr(syn, "_plugin_state", {}), "get", lambda *_: 0.01)("sigma", 0.01))
        torch = getattr(syn, "_torch", None)
        dev = getattr(syn, "_device", "cpu")
        if torch is not None:
            try:
                vt = value
                if not (hasattr(vt, "detach") and hasattr(vt, "to")):
                    vt = torch.tensor(self._to_list(syn, value), dtype=torch.float32, device=dev)
                noise = torch.randn_like(vt) * float(max(0.0, sigma))
                noisy = vt + noise
                return Synapse.transmit(syn, noisy, direction=direction)  # type: ignore[misc]
            except Exception:
                pass
        # Fallback: Python lists
        vals = self._to_list(syn, value)
        import random as _r
        noisy = [v + (sigma * (_r.random() * 2.0 - 1.0)) for v in vals]
        return Synapse.transmit(syn, noisy, direction=direction)  # type: ignore[misc]


__all__ = ["NoisySynapsePlugin"]
