from __future__ import annotations

"""Neuron plugin that learns to delegate to other neuron types.

`AutoNeuronPlugin` acts as a meta neuron type. For every forward pass it
chooses a concrete neuron type via a self-attention score over training
metrics exposed through ``expose_learnable_params`` and delegates execution to
that type. If the chosen type raises an exception (for example due to unmet
wiring requirements), the plugin rolls back to the previously successful type
and retries the step so training can continue without breaking gradients.
"""

import math
from typing import Any, List, Optional, Sequence

from ..wanderer import expose_learnable_params
from ..graph import _NEURON_TYPES


class AutoNeuronPlugin:
    """Meta neuron type that dynamically selects other neuron plugins.

    Parameters
    ----------
    disabled_types:
        Optional list of neuron type names that must never be selected.
        Types appearing in this list are excluded from the candidate set,
        preventing accidental delegation to undesired plugins.
    """

    def __init__(self, disabled_types: Optional[Sequence[str]] = None) -> None:
        self._current_neuron_id = 0
        self._disabled = set(disabled_types or [])

    # ---- Selection -----------------------------------------------------
    def _candidate_types(self) -> List[Optional[str]]:
        names: List[Optional[str]] = [None]
        names += [
            n
            for n in _NEURON_TYPES.keys()
            if n != "autoneuron" and n not in self._disabled
        ]
        return names

    def _select_type(self, wanderer: "Wanderer", neuron: "Neuron") -> Optional[str]:
        torch = getattr(wanderer, "_torch", None)
        device = getattr(wanderer, "_device", "cpu")
        names = self._candidate_types()
        base_score = self._attention_score(wanderer)
        scores: List[Any] = []
        for nm in names:
            key = nm or "base"
            bias_name = f"autoneuron_bias_{key}"
            gain_name = f"autoneuron_gain_{key}"
            try:
                wanderer.ensure_learnable_param(bias_name, 0.0)
                bias = wanderer.get_learnable_param_tensor(bias_name)
            except Exception:
                bias = 0.0
            try:
                wanderer.ensure_learnable_param(gain_name, 1.0)
                gain = wanderer.get_learnable_param_tensor(gain_name)
            except Exception:
                gain = 1.0
            try:
                score = base_score * gain + bias
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
    def _attention_score(
        self,
        wanderer: "Wanderer",
        *,
        q_loss: float = 1.0,
        q_speed: float = 0.5,
        q_complexity: float = 0.1,
        k_loss: float = 1.0,
        k_speed: float = 0.5,
        k_complexity: float = 0.1,
        v_loss: float = 1.0,
        v_speed: float = 0.5,
        v_complexity: float = 0.1,
    ):
        """Self-attention score over loss, speed and model size."""

        torch = getattr(wanderer, "_torch", None)
        loss = float(getattr(wanderer, "_last_walk_mean_loss", 0.0) or 0.0)
        steps = float(getattr(wanderer, "_walk_step_count", 0.0))
        brain = getattr(wanderer, "brain", None)
        complexity = float(
            len(getattr(brain, "neurons", {})) + len(getattr(brain, "synapses", []))
            if brain is not None
            else 0.0
        )
        if torch is None:
            scores = [
                q_loss * k_loss * loss,
                q_speed * k_speed * steps,
                q_complexity * k_complexity * complexity,
            ]
            m = max(scores)
            exps = [math.exp(s - m) for s in scores]
            denom = sum(exps) or 1.0
            weights = [e / denom for e in exps]
            vals = [v_loss * loss, v_speed * steps, v_complexity * complexity]
            return sum(w * v for w, v in zip(weights, vals))
        device = getattr(wanderer, "_device", "cpu")
        metrics = torch.tensor([loss, steps, complexity], dtype=torch.float32, device=device)

        def _to_tensor(val: Any) -> Any:
            return val if hasattr(val, "to") else torch.tensor(float(val), dtype=torch.float32, device=device)

        Q = torch.stack([_to_tensor(q_loss), _to_tensor(q_speed), _to_tensor(q_complexity)])
        K = torch.stack([_to_tensor(k_loss), _to_tensor(k_speed), _to_tensor(k_complexity)])
        V = torch.stack([_to_tensor(v_loss), _to_tensor(v_speed), _to_tensor(v_complexity)])
        scores = (metrics * Q) * (metrics * K)
        weights = torch.softmax(scores / math.sqrt(3.0), dim=0)
        return torch.sum(weights * (metrics * V))

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

