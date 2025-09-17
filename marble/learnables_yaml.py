"""Central registry and YAML sync for learnable parameters.

This module keeps track of every :class:`~marble.learnable_param.LearnableParam`
that is created via the different ``ensure_learnable_param`` helpers scattered
throughout the code base.  The registry is capable of:

* Recording which component owns the learnable and what its internal storage
  name is (``Wanderer``, ``Brain`` or ``SelfAttention`` variants).
* Providing a ``updatelearnablesyaml`` function that synchronises the
  ``learnables.yaml`` file with the currently known learnables.  The YAML file
  acts as both documentation and control surface – toggling a learnable to
  ``ON`` in the file enables optimisation for that parameter, while ``OFF``
  disables optimisation.

The implementation purposely avoids any heavy run‑time dependencies.  PyYAML is
imported lazily so simply importing :mod:`marble.learnables_yaml` has no side
effects other than creating the registry singleton.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
import importlib
import inspect
import pkgutil
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Set, Tuple
import weakref

from .learnable_param import LearnableParam


@dataclass
class _TrackedLearnable:
    """Internal bookkeeping entry for a single learnable parameter."""

    display_name: str
    internal_name: str
    owner_kind: str
    scope: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    owner_ref: weakref.ReferenceType[Any] = field(default_factory=lambda: lambda: None)
    param_ref: weakref.ReferenceType[LearnableParam] = field(default_factory=lambda: lambda: None)


class LearnableRegistry:
    """Tracks all learnable parameters that are registered at runtime."""

    def __init__(self) -> None:
        self._records: Dict[str, _TrackedLearnable] = {}
        self._by_param: Dict[int, str] = {}

    # -- registration -------------------------------------------------
    def register(
        self,
        owner: Any,
        internal_name: str,
        param: LearnableParam,
        *,
        display_name: Optional[str] = None,
        scope: str,
        metadata: Optional[Dict[str, Any]] = None,
        yaml_path: Optional[Path] = None,
    ) -> str:
        """Register *param* and ensure previously configured OPT state is applied.

        Parameters
        ----------
        owner:
            Component that stores the learnable (``Wanderer``, ``Brain`` or
            ``SelfAttention`` instance).
        internal_name:
            Key used by the owner to store the learnable.
        param:
            :class:`LearnableParam` instance to track.
        display_name:
            Human readable label used inside ``learnables.yaml``.  Defaults to
            ``internal_name`` when omitted.
        scope:
            String describing where the learnable lives.  Used for sanity checks
            when computing whether it is truly "ON".
        metadata:
            Arbitrary extra information (e.g. neuron ids for self-attention
            parameters).
        yaml_path:
            Optional override for the path to ``learnables.yaml`` when applying
            stored user preferences.
        """

        key = display_name or internal_name
        owner_kind = owner.__class__.__name__ if owner is not None else "Unknown"
        owner_ref: weakref.ReferenceType[Any]
        if owner is None:
            owner_ref = lambda: None  # type: ignore[assignment]
        else:
            try:
                owner_ref = weakref.ref(owner)
            except TypeError:
                # Fallback for objects without weakref support.
                owner_ref = lambda: owner  # type: ignore[assignment]

        record = _TrackedLearnable(
            display_name=key,
            internal_name=internal_name,
            owner_kind=owner_kind,
            scope=scope,
            metadata=dict(metadata or {}),
            owner_ref=owner_ref,
            param_ref=weakref.ref(param),
        )
        associated = _LEARNABLE_PLUGIN_ASSOC.get((owner_kind, internal_name))
        if associated:
            plugins = record.metadata.setdefault("plugins", [])
            for plugin_name in sorted(associated):
                if plugin_name not in plugins:
                    plugins.append(plugin_name)
        self._records[key] = record
        self._by_param[id(param)] = key
        try:
            param.display_name = key
        except Exception:
            pass

        # Apply persisted preference (if any) so the learnable immediately obeys
        # the ON/OFF state stored in learnables.yaml.
        desired_state = self._load_preferences(yaml_path).get(key)
        if desired_state == "ON":
            param.opt = True
        elif desired_state == "OFF":
            param.opt = False

        return key

    def has_active_plugin_learnable(self, plugin_class: str) -> bool:
        """Return ``True`` if any learnable linked to *plugin_class* is ON."""

        if not plugin_class:
            return False
        for _key, record, param, owner in self._iter_alive():
            plugins = record.metadata.get("plugins", [])
            if plugin_class not in plugins:
                continue
            if not getattr(param, "opt", False):
                continue
            if not self._is_hooked(owner, record, param):
                continue
            return True
        return False

    # -- inspection helpers ------------------------------------------
    def _iter_alive(self) -> Iterable[tuple[str, _TrackedLearnable, LearnableParam, Any]]:
        dead_keys = []
        for key, record in self._records.items():
            owner = record.owner_ref()
            param = record.param_ref()
            if owner is None or param is None:
                dead_keys.append(key)
                continue
            yield key, record, param, owner
        for key in dead_keys:
            self._records.pop(key, None)

    def log_owner_learnables(self, owner: Any) -> None:
        """Emit TensorBoard entries for all active learnables owned by *owner*."""

        try:
            from .reporter import report
        except Exception:
            return

        for key, record, param, current_owner in self._iter_alive():
            if current_owner is not owner:
                continue
            if not getattr(param, "opt", False):
                continue
            if not self._is_hooked(current_owner, record, param):
                continue
            value = self._prepare_log_value(param.tensor)
            if value is None:
                continue
            try:
                report("learnables", key, value)
            except Exception:
                continue

    # -- YAML synchronisation ----------------------------------------
    def update_yaml(self, yaml_path: Optional[Path] = None) -> None:
        """Synchronise ``learnables.yaml`` with the current registry state."""

        config = self._load_preferences(yaml_path)
        active_keys = set()
        for key, record, param, owner in self._iter_alive():
            active_keys.add(key)
            desired_state = config.get(key)
            desired_bool = desired_state == "ON"
            if desired_state is None:
                desired_bool = False
            if param.opt != desired_bool:
                param.opt = desired_bool
            hooked = self._is_hooked(owner, record, param)
            state = "ON" if (param.opt and hooked) else "OFF"
            config[key] = state

        # Remove stale entries for learnables that no longer exist.
        stale = set(config) - active_keys
        for key in stale:
            config.pop(key, None)

        self._write_preferences(config, yaml_path)

    def bulk_set_state(
        self,
        *,
        state: str,
        yaml_path: Optional[Path] = None,
        predicate: Optional[Callable[[str], bool]] = None,
    ) -> None:
        """Set the ON/OFF state for all learnables that match *predicate*."""

        target_state = "ON" if state.upper() == "ON" else "OFF"
        desired_bool = target_state == "ON"
        config = self._load_preferences(yaml_path)

        for key in list(config):
            if predicate is not None and not predicate(key):
                continue
            config[key] = target_state
            record = self._records.get(key)
            if record is None:
                continue
            param = record.param_ref()
            if param is None:
                continue
            param.opt = desired_bool

        self._write_preferences(config, yaml_path)

    # -- internals ----------------------------------------------------
    @staticmethod
    def default_yaml_path() -> Path:
        return Path(__file__).resolve().parents[1] / "learnables.yaml"

    def _prepare_log_value(self, raw: Any) -> Any:
        if raw is None:
            return None
        try:
            import torch  # type: ignore
        except Exception:
            torch = None  # type: ignore
        if torch is not None and isinstance(raw, torch.Tensor):
            try:
                return raw.detach().to("cpu")
            except Exception:
                try:
                    return raw.detach()
                except Exception:
                    return None
        try:
            detached = raw.detach()  # type: ignore[attr-defined]
            if hasattr(detached, "to"):
                try:
                    return detached.to("cpu")
                except Exception:
                    return detached
            return detached
        except Exception:
            pass
        if hasattr(raw, "cpu"):
            try:
                return raw.cpu()
            except Exception:
                pass
        if hasattr(raw, "tolist"):
            try:
                return raw.tolist()
            except Exception:
                pass
        return raw

    def _load_preferences(self, yaml_path: Optional[Path]) -> Dict[str, str]:
        from collections import OrderedDict

        path = yaml_path or self.default_yaml_path()
        if not path.exists():
            return OrderedDict()
        try:
            import yaml  # type: ignore

            with path.open("r", encoding="utf8") as fh:
                data = yaml.safe_load(fh) or {}
        except Exception:
            return OrderedDict()
        if not isinstance(data, dict):
            return OrderedDict()
        out: Dict[str, str] = OrderedDict()
        for key, value in data.items():
            if value is None:
                continue
            norm = str(value).strip().upper()
            out[str(key)] = "ON" if norm == "ON" else "OFF"
        return out

    def _write_preferences(self, config: Dict[str, str], yaml_path: Optional[Path]) -> None:
        try:
            import yaml  # type: ignore

            path = yaml_path or self.default_yaml_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            ordered = {k: config[k] for k in sorted(config)}
            with path.open("w", encoding="utf8") as fh:
                yaml.safe_dump(ordered, fh, sort_keys=False)
        except Exception:
            # Writing the file is best effort – never break execution if the
            # output path is not writable.
            pass

    def _is_hooked(self, owner: Any, record: _TrackedLearnable, param: LearnableParam) -> bool:
        scope = record.scope
        name = record.internal_name
        try:
            if scope == "wanderer":
                store = getattr(owner, "_learnables", {})
                return store.get(name) is param
            if scope == "brain":
                store = getattr(owner, "_learnables", {})
                return store.get(name) is param
            if scope == "selfattention_global":
                store = getattr(owner, "_global_learnables", {})
                if store.get(name) is not param:
                    return False
                return getattr(owner, "_owner", None) is not None
            if scope == "selfattention_neuron":
                nid = record.metadata.get("neuron_id")
                container = getattr(owner, "_learnables", {}).get(nid, {})
                if container.get(name) is not param:
                    return False
                return getattr(owner, "_owner", None) is not None
        except Exception:
            return False
        return True


_LEARNABLE_PLUGIN_ASSOC: Dict[Tuple[str, str], Set[str]] = defaultdict(set)


def register_learnable_plugin(owner_kind: str, internal_name: str, plugin_class: str) -> None:
    """Record that *internal_name* is associated with *plugin_class*.

    Parameters
    ----------
    owner_kind:
        Name of the class that owns the learnable (e.g. ``"Wanderer"``).
    internal_name:
        Key under which the learnable is stored on the owner.
    plugin_class:
        Name of the plugin or routine class that depends on the learnable.
    """

    if not owner_kind or not internal_name or not plugin_class:
        return
    key = (owner_kind, internal_name)
    _LEARNABLE_PLUGIN_ASSOC[key].add(plugin_class)
    for record in list(_REGISTRY._records.values()):
        if record.owner_kind != owner_kind:
            continue
        if record.internal_name != internal_name:
            continue
        plugins = record.metadata.setdefault("plugins", [])
        if plugin_class not in plugins:
            plugins.append(plugin_class)


_REGISTRY = LearnableRegistry()


_STATIC_DISCOVERY_PERFORMED = False


def _try_import_torch() -> Any:
    try:
        import torch  # type: ignore

        return torch
    except Exception:
        return None


def _make_learnable_param(
    init_value: Any,
    *,
    requires_grad: bool = True,
    lr: Optional[float] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> LearnableParam:
    torch_mod = _try_import_torch()
    tensor = init_value
    if torch_mod is not None:
        try:
            if hasattr(init_value, "detach"):
                tensor = init_value.detach().clone().to(dtype=torch_mod.float32, device="cpu")
            else:
                tensor = torch_mod.tensor(init_value, dtype=torch_mod.float32, device="cpu")
            if requires_grad and hasattr(tensor, "requires_grad_"):
                tensor.requires_grad_()
        except Exception:
            try:
                tensor = torch_mod.tensor([float(init_value)], dtype=torch_mod.float32, device="cpu")
                if requires_grad and hasattr(tensor, "requires_grad_"):
                    tensor.requires_grad_()
            except Exception:
                tensor = init_value
    return LearnableParam(
        tensor=tensor,
        orig_type=type(init_value),
        opt=False,
        lr=lr,
        min_value=min_value,
        max_value=max_value,
    )


def _create_wanderer_stub(brain_stub: Any) -> Any:
    torch_mod = _try_import_torch()

    def _ensure(self: Any, name: str, init_value: Any, **kwargs: Any) -> Any:
        if name in self._learnables:
            return self._learnables[name].tensor
        lp = _make_learnable_param(
            init_value,
            requires_grad=kwargs.get("requires_grad", True),
            lr=kwargs.get("lr"),
            min_value=kwargs.get("min_value"),
            max_value=kwargs.get("max_value"),
        )
        _REGISTRY.register(
            self,
            name,
            lp,
            display_name=f"Wanderer.{name}",
            scope="wanderer",
            metadata={},
        )
        self._learnables[name] = lp
        return lp.tensor

    def _get(self: Any, name: str) -> Any:
        ent = self._learnables.get(name)
        return None if ent is None else ent.tensor

    def _set_opt(self: Any, name: str, *, enabled: bool = True, lr: Optional[float] = None) -> None:
        ent = self._learnables.get(name)
        if ent is None:
            return
        ent.opt = bool(enabled)
        if lr is not None:
            try:
                ent.lr = float(lr)
            except Exception:
                pass

    def _init(self: Any) -> None:
        self._learnables = {}
        self.brain = brain_stub
        self._torch = torch_mod
        self._device = "cpu"
        self._plugin_state = {}
        self._wplugins = []
        self._neuro_plugins = []
        self._selfattentions = []
        self._pending_settings = {}
        self._last_walk_mean_loss = 0.0
        self._walk_step_count = 0

    attrs = {
        "__slots__": ("_learnables", "brain", "_torch", "_device", "_plugin_state", "_wplugins", "_neuro_plugins", "_selfattentions", "_pending_settings", "_last_walk_mean_loss", "_walk_step_count"),
        "__init__": _init,
        "ensure_learnable_param": _ensure,
        "get_learnable_param_tensor": _get,
        "set_param_optimization": _set_opt,
    }
    return type("Wanderer", (), attrs)()


def _create_brain_stub() -> Any:

    def _ensure(self: Any, name: str, init_value: Any, **kwargs: Any) -> Any:
        if name in self._learnables:
            return self._learnables[name].tensor
        lp = _make_learnable_param(
            init_value,
            requires_grad=kwargs.get("requires_grad", True),
            lr=kwargs.get("lr"),
            min_value=kwargs.get("min_value"),
            max_value=kwargs.get("max_value"),
        )
        _REGISTRY.register(
            self,
            name,
            lp,
            display_name=f"Brain.{name}",
            scope="brain",
            metadata={},
        )
        self._learnables[name] = lp
        return lp.tensor

    def _get(self: Any, name: str) -> Any:
        ent = self._learnables.get(name)
        return None if ent is None else ent.tensor

    def _set_opt(self: Any, name: str, *, enabled: bool = True, lr: Optional[float] = None) -> None:
        ent = self._learnables.get(name)
        if ent is None:
            return
        ent.opt = bool(enabled)
        if lr is not None:
            try:
                ent.lr = float(lr)
            except Exception:
                pass

    def _add_neuron(self: Any, index: Any, **kwargs: Any) -> Any:
        neuron = type(
            "Neuron",
            (),
            {
                "_plugin_state": {"wanderer": kwargs.get("wanderer")},
                "_ensure_tensor": lambda self, val: val,
                "tensor": kwargs.get("tensor"),
                "weight": kwargs.get("weight", 1.0),
                "bias": kwargs.get("bias", 0.0),
                "type_name": kwargs.get("type_name"),
            },
        )()
        key = tuple(index) if not isinstance(index, tuple) else index
        self.neurons[key] = neuron
        return neuron

    def _get_neuron(self: Any, index: Any) -> Any:
        key = tuple(index) if not isinstance(index, tuple) else index
        return self.neurons.get(key)

    def _init(self: Any) -> None:
        self._learnables = {}
        self.mode = "grid"
        self._device = "cpu"
        self.neurons = {}
        self.synapses = []
        self.wanderer = None

    attrs = {
        "__slots__": ("_learnables", "mode", "_device", "neurons", "synapses", "wanderer"),
        "__init__": _init,
        "ensure_learnable_param": _ensure,
        "get_learnable_param_tensor": _get,
        "set_param_optimization": _set_opt,
        "add_neuron": _add_neuron,
        "get_neuron": _get_neuron,
    }
    return type("Brain", (), attrs)()


def _call_with_owner(func: Any, *, wanderer: Any, brain: Any) -> None:
    try:
        sig = inspect.signature(func)
    except Exception:
        with suppress(Exception):
            func()
        return

    args = []
    kwargs: Dict[str, Any] = {}

    for name, param in sig.parameters.items():
        if name in {"self", "cls"}:
            continue
        owner = None
        if name == "wanderer":
            owner = wanderer
        elif name == "brain":
            owner = brain
        if owner is not None:
            if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                args.append(owner)
            else:
                kwargs[name] = owner
            continue
        if param.kind in {param.VAR_POSITIONAL, param.VAR_KEYWORD}:
            continue
        if param.default is inspect._empty:
            placeholder: Any = None
            if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                args.append(placeholder)
            else:
                kwargs[name] = placeholder

    with suppress(Exception):
        func(*args, **kwargs)


def _invoke_exposed_callables(wanderer: Any, brain: Any) -> None:
    seen: set[int] = set()
    for module in list(sys.modules.values()):
        name = getattr(module, "__name__", "")
        if not name.startswith("marble"):
            continue
        for attr in dir(module):
            with suppress(Exception):
                obj = getattr(module, attr)
                _maybe_invoke(obj, wanderer=wanderer, brain=brain, seen=seen)


def _maybe_invoke(obj: Any, *, wanderer: Any, brain: Any, seen: set[int]) -> None:
    if inspect.isclass(obj):
        instance = None
        with suppress(Exception):
            instance = obj()
        for name, value in vars(obj).items():
            if isinstance(value, staticmethod):
                candidate = value.__func__
            elif isinstance(value, classmethod):
                candidate = value.__func__
            else:
                candidate = value
            if not callable(candidate):
                continue
            if not getattr(candidate, "__exposes_learnables__", False):
                continue
            key = id(getattr(candidate, "__wrapped__", candidate))
            if key in seen:
                continue
            seen.add(key)
            bound = getattr(instance, name, None)
            if bound is None:
                bound = getattr(obj, name)
            _call_with_owner(bound, wanderer=wanderer, brain=brain)
        return

    if not callable(obj):
        return
    if not getattr(obj, "__exposes_learnables__", False):
        return
    target = getattr(obj, "__wrapped__", obj)
    key = id(target)
    if key in seen:
        return
    seen.add(key)
    _call_with_owner(obj, wanderer=wanderer, brain=brain)


def _hydrate_autoplugin(wanderer: Any) -> None:
    with suppress(Exception):
        from .plugins.wanderer_autoplugin import AutoPlugin
        from .wanderer import NEURO_TYPES_REGISTRY, WANDERER_TYPES_REGISTRY
        from .selfattention import _SELFA_TYPES  # type: ignore[attr-defined]
        from .buildingblock import _BUILDINGBLOCK_TYPES  # type: ignore[attr-defined]

        wanderer._wplugins = [type(name, (), {})() for name in sorted(WANDERER_TYPES_REGISTRY)]
        wanderer._neuro_plugins = [type(name, (), {})() for name in sorted(NEURO_TYPES_REGISTRY)]
        sa_stub = type(
            "SelfAttentionStub",
            (),
            {
                "_routines": [type(name, (), {})() for name in sorted(getattr(_SELFA_TYPES, "keys", lambda: [])())],
            },
        )()
        wanderer._selfattentions = [sa_stub]
        plugin = AutoPlugin()
        plugin.on_init(wanderer)
        plugin.before_walk(wanderer, None)


def _hydrate_autoneuron(wanderer: Any) -> None:
    with suppress(Exception):
        from .plugins.autoneuron import AutoNeuronPlugin

        plugin = AutoNeuronPlugin()
        neuron = type(
            "Neuron",
            (),
            {"_plugin_state": {"wanderer": wanderer}, "_ensure_tensor": lambda self, val: val},
        )()
        plugin._select_type(wanderer, neuron)  # type: ignore[attr-defined]


def _hydrate_latent_space(wanderer: Any) -> None:
    with suppress(Exception):
        from .plugins.wanderer_latentspace import LatentSpacePlugin

        plugin = LatentSpacePlugin()
        plugin.on_init(wanderer)


def _hydrate_neuron_builder(wanderer: Any) -> None:
    with suppress(Exception):
        from .plugins.wanderer_neuronbuilder import DynamicNeuronBuilderPlugin

        plugin = DynamicNeuronBuilderPlugin()
        plugin.on_walk_end(wanderer, {}, build_threshold=0.0)


def _perform_static_discovery() -> None:
    global _STATIC_DISCOVERY_PERFORMED
    if _STATIC_DISCOVERY_PERFORMED:
        return
    _STATIC_DISCOVERY_PERFORMED = True

    try:
        plugin_pkg = importlib.import_module("marble.plugins")
    except Exception:
        return

    for base in ("marble.marblemain", "marble.wanderer", "marble.selfattention"):
        with suppress(Exception):
            importlib.import_module(base)

    for module in pkgutil.walk_packages(getattr(plugin_pkg, "__path__", []), prefix="marble.plugins."):
        with suppress(Exception):
            importlib.import_module(module.name)

    brain_stub = _create_brain_stub()
    wanderer_stub = _create_wanderer_stub(brain_stub)
    brain_stub.wanderer = wanderer_stub  # type: ignore[attr-defined]

    _invoke_exposed_callables(wanderer_stub, brain_stub)
    _hydrate_autoplugin(wanderer_stub)
    _hydrate_autoneuron(wanderer_stub)
    _hydrate_latent_space(wanderer_stub)
    _hydrate_neuron_builder(wanderer_stub)


def register_learnable(
    owner: Any,
    internal_name: str,
    param: LearnableParam,
    *,
    display_name: Optional[str] = None,
    scope: str,
    metadata: Optional[Dict[str, Any]] = None,
    yaml_path: Optional[Path] = None,
) -> str:
    """Register *param* with the global :class:`LearnableRegistry`."""

    return _REGISTRY.register(
        owner,
        internal_name,
        param,
        display_name=display_name,
        scope=scope,
        metadata=metadata,
        yaml_path=yaml_path,
    )


def updatelearnablesyaml(yaml_path: Optional[Path] = None) -> None:
    """Synchronise ``learnables.yaml`` with the currently tracked learnables."""

    _perform_static_discovery()
    _REGISTRY.update_yaml(yaml_path=yaml_path)


def learnablesON(yaml_path: Optional[Path] = None) -> None:
    """Turn optimisation ON for all learnables that are not loss-related."""

    updatelearnablesyaml(yaml_path)
    _REGISTRY.bulk_set_state(
        state="ON",
        yaml_path=yaml_path,
        predicate=lambda key: "loss" not in key.lower(),
    )


def learnablesOFF(yaml_path: Optional[Path] = None) -> None:
    """Turn optimisation OFF for all learnables."""

    updatelearnablesyaml(yaml_path)
    _REGISTRY.bulk_set_state(state="OFF", yaml_path=yaml_path)


def log_learnable_values(owner: Any) -> None:
    """Log active learnables for *owner* into the reporter's ``learnables`` group."""

    _REGISTRY.log_owner_learnables(owner)


def plugin_learnable_forces_activation(plugin_class: str) -> bool:
    """Return ``True`` when any learnable tied to *plugin_class* is enabled."""

    return _REGISTRY.has_active_plugin_learnable(plugin_class)


__all__ = [
    "register_learnable",
    "updatelearnablesyaml",
    "learnablesON",
    "learnablesOFF",
    "LearnableRegistry",
    "log_learnable_values",
    "register_learnable_plugin",
    "plugin_learnable_forces_activation",
]

