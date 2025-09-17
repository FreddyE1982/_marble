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
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
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
        self._records[key] = record
        self._by_param[id(param)] = key

        # Apply persisted preference (if any) so the learnable immediately obeys
        # the ON/OFF state stored in learnables.yaml.
        desired_state = self._load_preferences(yaml_path).get(key)
        if desired_state == "ON":
            param.opt = True
        elif desired_state == "OFF":
            param.opt = False

        return key

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

    # -- internals ----------------------------------------------------
    @staticmethod
    def default_yaml_path() -> Path:
        return Path(__file__).resolve().parents[1] / "learnables.yaml"

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


_REGISTRY = LearnableRegistry()


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

    _REGISTRY.update_yaml(yaml_path=yaml_path)


__all__ = ["register_learnable", "updatelearnablesyaml", "LearnableRegistry"]

