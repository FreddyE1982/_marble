from __future__ import annotations

"""Wanderer plugin that learns to enable or disable other plugins.

The plugin wraps all existing Wanderer and neuroplasticity plugins with a
gating function. The gating decision is driven by learnable parameters exposed
via :func:`expose_learnable_params` and per-plugin bias terms registered on the
owning :class:`~marble.wanderer.Wanderer` instance. For every step the gate
considers the current walk step and the active neuron.

Learned objective prioritizes overall accuracy, then training speed, followed by
model size and complexity (implicitly via gradients adjusting the gate biases).
"""

from typing import Any, List, Tuple

from ..wanderer import expose_learnable_params


class AutoPlugin:
    """Meta-plugin that toggles other plugins on a per-step basis."""

    def __init__(self) -> None:
        self._neuro_wrapped = False
        self._current_neuron_id = 0

    def on_init(self, wanderer: "Wanderer") -> None:  # noqa: D401
        """Wrap existing Wanderer plugins with gating proxies."""

        wplugins = getattr(wanderer, "_wplugins", [])
        explicit = getattr(wanderer, "_explicit_wplugin_names", set())
        for i, p in enumerate(list(wplugins)):
            if p is self:
                continue
            name = p.__class__.__name__
            if name in explicit:
                continue
            wanderer.ensure_learnable_param(f"autoplugin_bias_{name}", 0.0)
            wplugins[i] = _GatedPlugin(p, name, self)

    def before_walk(self, wanderer: "Wanderer", start: "Neuron") -> None:
        """Ensure neuroplasticity plugins are wrapped before training."""

        if self._neuro_wrapped:
            return
        nplugins = getattr(wanderer, "_neuro_plugins", [])
        explicit_n = getattr(wanderer, "_explicit_neuroplugin_names", set())
        for i, p in enumerate(list(nplugins)):
            name = p.__class__.__name__
            if name in explicit_n:
                continue
            wanderer.ensure_learnable_param(f"autoplugin_bias_{name}", 0.0)
            nplugins[i] = _GatedPlugin(p, name, self)
        self._neuro_wrapped = True

    def is_active(
        self, wanderer: "Wanderer", name: str, neuron: "Neuron | None"
    ) -> bool:
        """Return True if the named plugin should be active."""

        self._current_neuron_id = 0 if neuron is None else id(neuron)
        explicit_w = getattr(wanderer, "_explicit_wplugin_names", set())
        explicit_n = getattr(wanderer, "_explicit_neuroplugin_names", set())
        if name in explicit_w or name in explicit_n:
            return True
        torch = getattr(wanderer, "_torch", None)
        if torch is None:
            return True
        bias = wanderer.get_learnable_param_tensor(f"autoplugin_bias_{name}")
        score = self._raw_score(wanderer)
        gate = torch.sigmoid(score + bias)
        return bool(gate.detach().to("cpu").item() > 0.5)

    @expose_learnable_params
    def _raw_score(
        self,
        wanderer: "Wanderer",
        *,
        base: float = 0.0,
        step_weight: float = 0.0,
        neuron_weight: float = 0.0,
    ):
        """Compute raw activation score prior to bias and sigmoid."""

        torch = getattr(wanderer, "_torch", None)
        if torch is None:
            return 0.0
        step = getattr(wanderer, "_walk_step_count", 0)
        nid = getattr(self, "_current_neuron_id", 0)
        return base + step_weight * step + neuron_weight * nid


class _GatedPlugin:
    """Proxy that consults :class:`AutoPlugin` before delegating."""

    def __init__(self, plugin: Any, name: str, controller: AutoPlugin) -> None:
        self._plugin = plugin
        self._name = name
        self._controller = controller

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - passthrough
        return getattr(self._plugin, item)

    def on_init(self, wanderer: "Wanderer") -> None:
        if hasattr(self._plugin, "on_init"):
            self._plugin.on_init(wanderer)

    def before_walk(self, wanderer: "Wanderer", start: "Neuron") -> None:
        if hasattr(self._plugin, "before_walk") and self._controller.is_active(
            wanderer, self._name, start
        ):
            self._plugin.before_walk(wanderer, start)

    def choose_next(
        self,
        wanderer: "Wanderer",
        current: "Neuron",
        choices: List[Tuple["Synapse", str]],
    ):
        if hasattr(self._plugin, "choose_next") and self._controller.is_active(
            wanderer, self._name, current
        ):
            return self._plugin.choose_next(wanderer, current, choices)
        return None

    def loss(self, wanderer: "Wanderer", outputs: List[Any]):
        if hasattr(self._plugin, "loss") and self._controller.is_active(
            wanderer, self._name, None
        ):
            return self._plugin.loss(wanderer, outputs)
        torch = getattr(wanderer, "_torch", None)
        return 0.0 if torch is None else torch.tensor(0.0, device=getattr(wanderer, "_device", "cpu"))


__all__ = ["AutoPlugin"]

