"""Runtime catalogues and telemetry for Marble plugins.

This module fulfils the Step 1.1 requirement from the improvement development
plan by collecting rich metadata about every plugin that is automatically
registered and by emitting reporter metrics that track activation frequency and
latency.  The telemetry is designed to power upcoming Mixture-of-Experts routing
logic, where we need detailed information about which plugins act as feature
extractors, meta-optimisers, path planners, or other functional niches.

Usage overview
--------------

``register_plugin_metadata`` is invoked from :mod:`marble.plugins` during the
auto-discovery pass.  It inspects the plugin implementation, infers a functional
niche via keyword heuristics (sourced from the architecture notes), and stores a
serialisable metadata snapshot.  ``record_plugin_activation`` is called by the
generic plugin dispatcher in :mod:`marble.marblemain` whenever a plugin hook is
executed.  The dispatcher supplies the plugin name, hook name, and measured
latency so we can update exponential statistics.

The collected information is exposed through ``get_plugin_catalog`` and
``get_plugin_usage`` for programmatic access and is mirrored into the central
reporter tree under ``plugins/metadata/catalog`` and ``plugins/metrics/usage``.
Tests can call ``reset_plugin_usage`` to clear metrics between scenarios.
"""

from __future__ import annotations

import inspect
import threading
import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, MutableMapping, Optional

from .reporter import report


def _load_architecture_roles() -> Dict[str, Dict[str, str]]:
    """Return plugin role hints extracted from ``ARCHITECTURE.md``.

    The architecture document enumerates canonical plugins together with a
    concise description of their responsibility.  Those summaries provide a
    higher-quality "functional niche" signal than simple keyword heuristics,
    especially for meta-plugins such as the new Mixture-of-Experts router.

    The loader scans bullet points of the form ``- Wanderer plugin `name`:`` and
    records the accompanying prose.  When multiple bullet points mention the
    same plugin we keep the longest description which typically corresponds to
    the most recent documentation update.
    """

    base = os.path.dirname(os.path.dirname(__file__))
    arch_path = os.path.join(base, "ARCHITECTURE.md")
    roles: Dict[str, Dict[str, str]] = {}
    if not os.path.exists(arch_path):
        return roles

    pattern = re.compile(
        r"-\s*(?P<label>[A-Za-z\-/ ]+?)\s+plugin\s+`(?P<name>[^`]+)`:\s*(?P<desc>.+)",
        re.IGNORECASE,
    )
    type_map = {
        "wanderer": "wanderer",
        "neuron": "neuron",
        "synapse": "synapse",
        "brain-train": "brain_train",
        "brain_train": "brain_train",
        "brain train": "brain_train",
        "selfattention": "selfattention",
        "self-attention": "selfattention",
        "neuroplasticity": "neuroplasticity",
        "buildingblock": "buildingblock",
        "building block": "buildingblock",
    }

    try:
        with open(arch_path, "r", encoding="utf-8") as handle:
            for raw in handle:
                match = pattern.search(raw)
                if not match:
                    continue
                label = (match.group("label") or "").strip().lower()
                plugin_name = match.group("name").strip()
                desc = match.group("desc").strip()
                plugin_type = None
                for key, mapped in type_map.items():
                    if key in label:
                        plugin_type = mapped
                        break
                entry = roles.get(plugin_name, {})
                if plugin_type:
                    entry["plugin_type"] = plugin_type
                # Prefer the most informative description (longest string)
                if len(desc) > len(entry.get("description", "")):
                    entry["description"] = desc
                roles[plugin_name] = entry
    except Exception:
        return roles

    return roles


_ARCHITECTURE_ROLES = _load_architecture_roles()


_DEFAULT_NICHES: Dict[str, str] = {
    "neuron": "activation_shaper",
    "synapse": "connectivity_modulator",
    "wanderer": "path_planner",
    "brain_train": "training_controller",
    "selfattention": "attention_moderator",
    "neuroplasticity": "plasticity_regulator",
    "buildingblock": "graph_editor",
}

_KEYWORD_NICHES: List[tuple[str, str]] = [
    ("conv", "feature_extractor"),
    ("pool", "feature_compactor"),
    ("fold", "feature_compactor"),
    ("unpool", "feature_expander"),
    ("attention", "attention_controller"),
    ("entropy", "stochastic_explorer"),
    ("quantum", "quantum_explorer"),
    ("optimizer", "meta_optimizer"),
    ("schedule", "schedule_controller"),
    ("memory", "memory_stabiliser"),
    ("loss", "loss_shaper"),
    ("builder", "structure_builder"),
    ("plastic", "plasticity_regulator"),
    ("synapse", "connectivity_modulator"),
    ("wander", "path_planner"),
    ("neuro", "plasticity_regulator"),
    ("train", "training_controller"),
]


@dataclass
class PluginMetadata:
    """Metadata snapshot describing a Marble plugin."""

    name: str
    plugin_type: str
    module: str
    class_name: str
    description: str
    niche: str
    plugin_id: Optional[int]
    hooks: List[str]
    architecture_role: Optional[str] = None

    def serialise(self) -> Dict[str, Any]:
        """Return a dict representation safe for reporting."""

        data = asdict(self)
        # Hooks are short names; keep them sorted for readability.
        data["hooks"] = sorted(set(self.hooks))
        return data


class _PluginCatalog:
    """In-memory catalogue of plugin metadata."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: Dict[str, PluginMetadata] = {}

    def register(
        self,
        name: str,
        plugin_type: str,
        plugin: Any,
        plugin_id: Optional[int] = None,
    ) -> PluginMetadata:
        """Register (or update) metadata for ``name``."""

        cls = plugin.__class__
        module = getattr(cls, "__module__", "")
        class_name = getattr(cls, "__name__", "")
        doc = inspect.getdoc(cls) or inspect.getdoc(plugin) or ""
        description = doc.splitlines()[0].strip() if doc else f"Plugin {class_name}"
        hooks = [
            attr
            for attr, value in inspect.getmembers(plugin)
            if not attr.startswith("_") and callable(value)
        ]
        text = f"{name} {module} {class_name} {doc}".lower()
        niche = _DEFAULT_NICHES.get(plugin_type, "generalist")
        for keyword, mapped in _KEYWORD_NICHES:
            if keyword in text:
                niche = mapped
                break
        meta = PluginMetadata(
            name=name,
            plugin_type=plugin_type,
            module=module,
            class_name=class_name,
            description=description,
            niche=niche,
            plugin_id=plugin_id,
            hooks=hooks,
            architecture_role=None,
        )
        arch = _ARCHITECTURE_ROLES.get(name)
        if arch:
            desc = arch.get("description")
            if desc:
                meta.architecture_role = desc
            if arch.get("plugin_type"):
                meta.plugin_type = arch["plugin_type"]
        with self._lock:
            self._entries[name] = meta
        self._report_single(meta)
        return meta

    def snapshot(self) -> Dict[str, PluginMetadata]:
        with self._lock:
            return dict(self._entries)

    def serialised_snapshot(self) -> Dict[str, Dict[str, Any]]:
        snap = self.snapshot()
        return {name: meta.serialise() for name, meta in snap.items()}

    def lookup(self, name: str) -> Optional[PluginMetadata]:
        with self._lock:
            return self._entries.get(name)

    def reset(self) -> None:
        with self._lock:
            self._entries.clear()
        report("plugins", "catalog", {}, "metadata")

    def _report_single(self, meta: PluginMetadata) -> None:
        try:
            report("plugins", "catalog", {meta.name: meta.serialise()}, "metadata")
        except Exception:
            pass


class _PluginUsage:
    """Aggregate per-plugin activation statistics."""

    def __init__(self, catalog: _PluginCatalog) -> None:
        self._catalog = catalog
        self._lock = threading.Lock()
        self._stats: Dict[str, Dict[str, Any]] = {}

    def record(self, name: str, hook_name: str, elapsed: float) -> None:
        hook = hook_name or "call"
        lat = max(float(elapsed), 0.0)
        hook_key = hook
        with self._lock:
            entry = self._stats.setdefault(
                name,
                {
                    "calls": 0,
                    "total_latency": 0.0,
                    "last_latency_ms": 0.0,
                    "hooks": {},
                    "plugin_type": None,
                },
            )
            entry["calls"] += 1
            entry["total_latency"] += lat
            entry["last_latency_ms"] = lat * 1_000.0
            meta = self._catalog.lookup(name)
            if meta is not None:
                entry["plugin_type"] = meta.plugin_type
            hooks: MutableMapping[str, Dict[str, Any]] = entry["hooks"]
            hook_stats = hooks.setdefault(
                hook_key,
                {"calls": 0, "total_latency": 0.0, "last_latency_ms": 0.0},
            )
            hook_stats["calls"] += 1
            hook_stats["total_latency"] += lat
            hook_stats["last_latency_ms"] = lat * 1_000.0
            entry_copy = self._format_entry(name, entry)
        self._report_usage(name, entry_copy)

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {name: self._format_entry(name, entry) for name, entry in self._stats.items()}

    def reset(self) -> None:
        with self._lock:
            self._stats.clear()
        report("plugins", "usage", {}, "metrics")

    def _format_entry(self, name: str, entry: Dict[str, Any]) -> Dict[str, Any]:
        calls = int(entry.get("calls", 0))
        total_latency = float(entry.get("total_latency", 0.0))
        avg_ms = (total_latency / calls * 1_000.0) if calls else 0.0
        hooks = {
            hook: {
                "calls": int(stats.get("calls", 0)),
                "avg_latency_ms": (
                    (float(stats.get("total_latency", 0.0)) / stats.get("calls", 1)) * 1_000.0
                    if stats.get("calls", 0)
                    else 0.0
                ),
                "last_latency_ms": float(stats.get("last_latency_ms", 0.0)),
            }
            for hook, stats in entry.get("hooks", {}).items()
        }
        return {
            "plugin": name,
            "plugin_type": entry.get("plugin_type"),
            "calls": calls,
            "avg_latency_ms": avg_ms,
            "last_latency_ms": float(entry.get("last_latency_ms", 0.0)),
            "hooks": hooks,
        }

    def _report_usage(self, name: str, entry: Dict[str, Any]) -> None:
        try:
            report("plugins", "usage", {name: entry}, "metrics")
        except Exception:
            pass


_CATALOG = _PluginCatalog()
_USAGE = _PluginUsage(_CATALOG)


def register_plugin_metadata(
    name: str,
    plugin_type: str,
    plugin: Any,
    *,
    plugin_id: Optional[int] = None,
) -> PluginMetadata:
    """Public helper for registering plugin metadata.

    Parameters
    ----------
    name:
        The public plugin identifier.
    plugin_type:
        One of ``{"neuron", "synapse", "wanderer", "brain_train",
        "selfattention", "neuroplasticity", "buildingblock"}``.
    plugin:
        The instantiated plugin object.
    plugin_id:
        Optional numeric identifier from the plugin discovery system.
    """

    return _CATALOG.register(name, plugin_type, plugin, plugin_id)


def record_plugin_activation(name: str, hook_name: str, elapsed: float) -> None:
    """Record a plugin activation with the measured elapsed time."""

    if not name:
        return
    _USAGE.record(name, hook_name, elapsed)


def get_plugin_catalog() -> Dict[str, Dict[str, Any]]:
    """Return a serialisable snapshot of the plugin catalogue."""

    return _CATALOG.serialised_snapshot()


def get_plugin_usage() -> Dict[str, Dict[str, Any]]:
    """Return aggregated plugin usage statistics."""

    return _USAGE.snapshot()


def reset_plugin_usage() -> None:
    """Clear accumulated usage metrics (primarily for tests)."""

    _USAGE.reset()


def reset_plugin_catalog() -> None:
    """Clear the plugin metadata catalogue."""

    _CATALOG.reset()


__all__ = [
    "PluginMetadata",
    "register_plugin_metadata",
    "record_plugin_activation",
    "get_plugin_catalog",
    "get_plugin_usage",
    "reset_plugin_usage",
    "reset_plugin_catalog",
]

