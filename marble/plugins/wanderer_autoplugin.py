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

from typing import Any, Dict, List, Tuple, Optional

from ..wanderer import expose_learnable_params
from ..buildingblock import get_buildingblock_type, _BUILDINGBLOCK_TYPES


class AutoPlugin:
    """Meta-plugin that toggles other plugins on a per-step basis.

    Parameters
    ----------
    disabled_plugins:
        Optional list of plugin class names that should be fully disabled.
        Any Wanderer or neuroplasticity plugin whose class name appears in
        this list will be removed from the active plugin stacks entirely.
    """

    def __init__(self, disabled_plugins: Optional[List[str]] = None) -> None:
        self._neuro_wrapped = False
        self._sa_wrapped = False
        self._current_neuron_id = 0
        self._disabled = set(disabled_plugins or [])

    def on_init(self, wanderer: "Wanderer") -> None:  # noqa: D401
        """Wrap existing Wanderer plugins with gating proxies."""

        wplugins = getattr(wanderer, "_wplugins", [])
        explicit = getattr(wanderer, "_explicit_wplugin_names", set())
        new_stack: List[Any] = []
        for p in list(wplugins):
            if p is self:
                new_stack.append(p)
                continue
            name = p.__class__.__name__
            if name in self._disabled:
                continue
            if name in explicit:
                new_stack.append(p)
                continue
            wanderer.ensure_learnable_param(f"autoplugin_bias_{name}", 0.0)
            new_stack.append(_GatedPlugin(p, name, self))
        wanderer._wplugins = new_stack
        for bb_name in _BUILDINGBLOCK_TYPES.keys():
            wanderer.ensure_learnable_param(f"autoplugin_bias_{bb_name}", 0.0)

    def before_walk(self, wanderer: "Wanderer", start: "Neuron") -> None:
        """Ensure all plugin stacks are wrapped before training."""

        if not self._neuro_wrapped:
            nplugins = getattr(wanderer, "_neuro_plugins", [])
            explicit_n = getattr(wanderer, "_explicit_neuroplugin_names", set())
            new_stack: List[Any] = []
            for p in list(nplugins):
                name = p.__class__.__name__
                if name in self._disabled:
                    continue
                if name in explicit_n:
                    new_stack.append(p)
                    continue
                wanderer.ensure_learnable_param(f"autoplugin_bias_{name}", 0.0)
                new_stack.append(_GatedPlugin(p, name, self))
            wanderer._neuro_plugins = new_stack
            self._neuro_wrapped = True

        if not self._sa_wrapped:
            for sa in getattr(wanderer, "_selfattentions", []) or []:
                routines = getattr(sa, "_routines", [])
                new_routines: List[Any] = []
                for r in list(routines):
                    if isinstance(r, _GatedSARoutine):
                        new_routines.append(r)
                        continue
                    name = r.__class__.__name__
                    if name in self._disabled:
                        continue
                    wanderer.ensure_learnable_param(f"autoplugin_bias_{name}", 0.0)
                    new_routines.append(_GatedSARoutine(r, name, self))
                sa._routines = new_routines
            self._sa_wrapped = True

    def is_active(
        self, wanderer: "Wanderer", name: str, neuron: "Neuron | None"
    ) -> bool:
        """Return True if the named plugin should be active."""

        self._current_neuron_id = 0 if neuron is None else id(neuron)
        if name in self._disabled:
            return False
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

    def apply_buildingblock(
        self, wanderer: "Wanderer", name: str, *args: Any, **kwargs: Any
    ) -> Any:
        """Apply a building block if its gate is active."""

        block = get_buildingblock_type(name)
        if block is None or name in self._disabled:
            return None
        wanderer.ensure_learnable_param(f"autoplugin_bias_{name}", 0.0)
        if not self.is_active(wanderer, name, None):
            return None
        return block.apply(wanderer.brain, *args, **kwargs)

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


class _GatedSARoutine:
    """Proxy for SelfAttention routines controlled by :class:`AutoPlugin`."""

    def __init__(self, routine: Any, name: str, controller: AutoPlugin) -> None:
        self._routine = routine
        self._name = name
        self._controller = controller

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - passthrough
        return getattr(self._routine, item)

    def on_init(self, selfattention: "SelfAttention") -> None:
        if hasattr(self._routine, "on_init"):
            self._routine.on_init(selfattention)

    def after_step(
        self,
        selfattention: "SelfAttention",
        reporter_ro: Any,
        wanderer: "Wanderer",
        step_index: int,
        ctx: Dict[str, Any],
    ):
        if hasattr(self._routine, "after_step") and self._controller.is_active(
            wanderer, self._name, None
        ):
            return self._routine.after_step(selfattention, reporter_ro, wanderer, step_index, ctx)
        return None


__all__ = ["AutoPlugin"]

