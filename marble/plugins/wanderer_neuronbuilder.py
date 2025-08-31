from __future__ import annotations

"""Wanderer plugin that learns to assemble building blocks into new neuron types.

`DynamicNeuronBuilderPlugin` attaches to a :class:`~marble.wanderer.Wanderer` and
uses a tiny self‑attention mechanism to decide how to combine low level
BuildingBlock plugins.  During ``on_walk_end`` it creates a fresh neuron,
optionally tweaks its weight and bias via attention‑selected blocks and then
registers a new neuron type so meta plugins like ``AutoNeuron`` can pick it up
automatically.
"""

from typing import Any, Dict

from ..wanderer import expose_learnable_params
from ..buildingblock import get_buildingblock_type
from ..graph import register_neuron_type


class DynamicNeuronBuilderPlugin:
    """Create new neuron types by attending over BuildingBlocks."""

    def __init__(self) -> None:
        self._create = None
        self._chg_weight = None
        self._chg_bias = None
        self._chg_type = None
        self._counter = 0

    @expose_learnable_params
    def on_walk_end(
        self,
        wanderer: "Wanderer",
        stats: Dict[str, Any],
        *,
        build_threshold: float = 0.5,
    ) -> None:
        """Create a new neuron at the end of each walk.

        Parameters
        ----------
        build_threshold:
            Attention score above which a BuildingBlock will be applied.
        """

        torch = getattr(wanderer, "_torch", None)
        if torch is None:
            return
        if self._create is None:
            self._create = get_buildingblock_type("create_neuron")
        if self._chg_weight is None:
            self._chg_weight = get_buildingblock_type("change_neuron_weight")
        if self._chg_bias is None:
            self._chg_bias = get_buildingblock_type("change_neuron_bias")
        if self._chg_type is None:
            self._chg_type = get_buildingblock_type("change_neuron_type")
        if self._create is None or self._chg_type is None:
            return
        device = getattr(wanderer, "_device", "cpu")

        # --- tiny self-attention over weight and bias blocks -----------------
        wanderer.ensure_learnable_param("dyn_query", [0.0, 0.0])
        wanderer.ensure_learnable_param("dyn_key_weight", [1.0, 0.0])
        wanderer.ensure_learnable_param("dyn_key_bias", [0.0, 1.0])
        q = wanderer.get_learnable_param_tensor("dyn_query")
        k_w = wanderer.get_learnable_param_tensor("dyn_key_weight")
        k_b = wanderer.get_learnable_param_tensor("dyn_key_bias")
        scores = torch.softmax(torch.stack([torch.dot(q, k_w), torch.dot(q, k_b)]), dim=0)
        w_prob, b_prob = scores[0], scores[1]

        wanderer.ensure_learnable_param("dyn_weight_val", 1.0)
        wanderer.ensure_learnable_param("dyn_bias_val", 0.0)
        weight_val = wanderer.get_learnable_param_tensor("dyn_weight_val") * w_prob
        bias_val = wanderer.get_learnable_param_tensor("dyn_bias_val") * b_prob

        if hasattr(build_threshold, "detach"):
            thr = float(build_threshold.detach().to("cpu").item())
        else:
            thr = float(build_threshold)

        brain = wanderer.brain
        idx = (self._counter,)
        tensor = torch.zeros(1, device=device)
        self._create.apply(brain, idx, tensor, weight=1.0, bias=0.0, type_name=None)

        if self._chg_weight is not None and float(w_prob.detach().to("cpu").item()) > thr:
            self._chg_weight.apply(brain, idx, weight=weight_val)
        if self._chg_bias is not None and float(b_prob.detach().to("cpu").item()) > thr:
            self._chg_bias.apply(brain, idx, bias=bias_val)

        type_name = f"dyn_{self._counter}"

        class GeneratedNeuronPlugin:
            def forward(self, neuron: "Neuron", input_value=None):
                x = neuron._ensure_tensor(neuron.tensor if input_value is None else input_value)
                return x * neuron.weight + neuron.bias

        register_neuron_type(type_name, GeneratedNeuronPlugin())
        self._chg_type.apply(brain, idx, type_name=type_name)
        self._counter += 1


__all__ = ["DynamicNeuronBuilderPlugin"]
