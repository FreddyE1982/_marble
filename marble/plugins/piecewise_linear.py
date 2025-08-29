from __future__ import annotations

"""Piecewise linear neuron plugin.

Implements two linear segments split at a learnable breakpoint. Each
segment has its own learnable slope and intercept. Parameters are
registered through ``expose_learnable_params`` so optimisation can shape
the piecewise function.
"""

from typing import Any, Tuple

from ..wanderer import expose_learnable_params


class PiecewiseLinearNeuronPlugin:
    """Apply a two-segment piecewise linear transform."""

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer: "Wanderer",
        *,
        pw_break: float = 0.0,
        pw_m1: float = 1.0,
        pw_c1: float = 0.0,
        pw_m2: float = 1.0,
        pw_c2: float = 0.0,
    ) -> Tuple[Any, Any, Any, Any, Any]:
        return pw_break, pw_m1, pw_c1, pw_m2, pw_c2

    def forward(self, neuron: "Neuron", input_value=None):
        x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)

        brk, m1, c1, m2, c2 = 0.0, 1.0, 0.0, 1.0, 0.0
        wanderer = neuron._plugin_state.get("wanderer")
        if wanderer is not None:
            try:
                brk, m1, c1, m2, c2 = self._params(wanderer)
            except Exception:
                pass

        torch = getattr(neuron, "_torch", None)
        if torch is not None and neuron._is_torch_tensor(x):
            y1 = m1 * x + c1
            y2 = m2 * x + c2
            cond = x < brk
            return torch.where(cond, y1, y2)

        x_list = x if isinstance(x, list) else [float(x)]
        brk_f = float(brk if not hasattr(brk, "detach") else brk.detach().to("cpu").item())
        m1_f = float(m1 if not hasattr(m1, "detach") else m1.detach().to("cpu").item())
        c1_f = float(c1 if not hasattr(c1, "detach") else c1.detach().to("cpu").item())
        m2_f = float(m2 if not hasattr(m2, "detach") else m2.detach().to("cpu").item())
        c2_f = float(c2 if not hasattr(c2, "detach") else c2.detach().to("cpu").item())
        out = [m1_f * v + c1_f if v < brk_f else m2_f * v + c2_f for v in map(float, x_list)]
        return out if len(out) != 1 else out[0]


__all__ = ["PiecewiseLinearNeuronPlugin"]

