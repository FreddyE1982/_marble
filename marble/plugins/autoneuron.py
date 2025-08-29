from __future__ import annotations

"""Neuron plugin that learns to delegate to other neuron types.

`AutoNeuronPlugin` acts as a meta neuron type. For every forward pass it
chooses a concrete neuron type via learnable scores exposed through
``expose_learnable_params`` and delegates execution to that type. If the
chosen type raises an exception (for example due to unmet wiring
requirements), the plugin rolls back to the previously successful type and
retries the step so training can continue without breaking gradients.
"""

from typing import Any, List, Optional

from ..wanderer import expose_learnable_params
from ..graph import _NEURON_TYPES


class AutoNeuronPlugin:
    """Meta neuron type that dynamically selects other neuron plugins."""

    def __init__(self) -> None:
        self._current_neuron_id = 0

    # ---- Selection -----------------------------------------------------
    def _candidate_types(self) -> List[Optional[str]]:
        names: List[Optional[str]] = [None]
        names += [n for n in _NEURON_TYPES.keys() if n != "autoneuron"]
        return names

    def _select_type(self, wanderer: "Wanderer", neuron: "Neuron") -> Optional[str]:
        torch = getattr(wanderer, "_torch", None)
        device = getattr(wanderer, "_device", "cpu")
        names = self._candidate_types()
        scores: List[Any] = []
        for nm in names:
            bias_name = f"autoneuron_bias_{nm or 'base'}"
            try:
                wanderer.ensure_learnable_param(bias_name, 0.0)
                bias = wanderer.get_learnable_param_tensor(bias_name)
            except Exception:
                bias = 0.0
            try:
                score = self._raw_score(wanderer) + bias
            except Exception:
                score = bias
            scores.append(score)
        if torch is None:
            return names[0]
        try:
            stacked = torch.stack(
                [
                    s if hasattr(s, "to") else torch.tensor(float(s), device=device)
                    for s in scores
                ]
            )
            idx = int(torch.argmax(stacked).detach().to("cpu").item())
            return names[idx]
        except Exception:
            return names[0]

    @expose_learnable_params
    def _raw_score(
        self,
        wanderer: "Wanderer",
        *,
        base: float = 0.0,
        step_weight: float = 0.0,
        neuron_weight: float = 0.0,
    ):
        """Compute a score using walk step and neuron id."""

        torch = getattr(wanderer, "_torch", None)
        if torch is None:
            return base
        step = getattr(wanderer, "_walk_step_count", 0)
        nid = getattr(self, "_current_neuron_id", 0)
        return base + step_weight * step + neuron_weight * nid

    # ---- Core operations -----------------------------------------------
    def _base_forward(self, neuron: "Neuron", input_value=None):
        x = neuron._ensure_tensor(
            neuron.tensor if input_value is None else input_value
        )
        torch = getattr(neuron, "_torch", None)
        if torch is not None and neuron._is_torch_tensor(x):
            return x * neuron.weight + neuron.bias
        xl = x if isinstance(x, list) else list(x)  # type: ignore[arg-type]
        return [neuron.weight * float(v) + neuron.bias for v in xl]

    def forward(self, neuron: "Neuron", input_value=None):
        wanderer = neuron._plugin_state.get("wanderer")
        prev = neuron._plugin_state.get("prev_type")
        chosen = prev
        if wanderer is not None:
            self._current_neuron_id = id(neuron)
            chosen = self._select_type(wanderer, neuron)
        plugin = _NEURON_TYPES.get(chosen) if chosen else None
        try:
            neuron.type_name = chosen
            if plugin is None:
                out = self._base_forward(neuron, input_value)
            else:
                out = plugin.forward(neuron, input_value)
            neuron._plugin_state["prev_type"] = chosen
            return out
        except Exception:
            neuron.type_name = prev
            plugin_prev = _NEURON_TYPES.get(prev) if prev else None
            if plugin_prev is None:
                out = self._base_forward(neuron, input_value)
            else:
                out = plugin_prev.forward(neuron, input_value)
            neuron._plugin_state["prev_type"] = prev
            return out
        finally:
            neuron.type_name = "autoneuron"

    def receive(self, neuron: "Neuron", value):
        prev = neuron._plugin_state.get("prev_type")
        plugin = _NEURON_TYPES.get(prev) if prev else None
        try:
            neuron.type_name = prev
            if plugin is not None and hasattr(plugin, "receive"):
                plugin.receive(neuron, value)
            else:
                neuron.tensor = neuron._ensure_tensor(value)
        finally:
            neuron.type_name = "autoneuron"


__all__ = ["AutoNeuronPlugin"]

