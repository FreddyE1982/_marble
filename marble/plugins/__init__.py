"""Plugin package auto-loader with automatic registration.

All modules inside this package are imported and inspected for plugin
classes. Modules are processed in name-sorted order so that registration is
deterministic. Any class whose name ends with ``Plugin`` or ``Routine`` is
registered automatically based on the module's name. This means dropping a
new module into :mod:`marble.plugins` immediately exposes the plugin without
any manual ``register_*`` calls.  As long as the set of plugin files remains
unchanged, every plugin retains the same numeric identifier across runs.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import List, Optional, Type, Any

from ..graph import register_neuron_type, register_synapse_type
from ..wanderer import register_wanderer_type, register_neuroplasticity_type
from ..selfattention import register_selfattention_type
from ..marblemain import register_brain_train_type
from ..buildingblock import register_buildingblock_type
from ..plugin_graph import PLUGIN_GRAPH

# Global registry assigning a unique numeric ID to every plugin.  The IDs are
# stable across runs as long as the set of available plugins does not change.
# This enables neural modules to embed plugins via ``nn.Embedding`` without any
# manual bookkeeping in individual plugin implementations.
PLUGIN_ID_REGISTRY: dict[str, int] = {}


def _assign_plugin_id(name: str, cls: type) -> int:
    """Return a stable integer ID for ``name`` and attach it to ``cls``.

    The assigned ID is stored both in :data:`PLUGIN_ID_REGISTRY` and as
    ``plugin_id``/``PLUGIN_ID`` attributes on the plugin class so that
    instantiated plugins can access it directly.
    """

    pid = PLUGIN_ID_REGISTRY.setdefault(name, len(PLUGIN_ID_REGISTRY))
    setattr(cls, "PLUGIN_ID", pid)
    setattr(cls, "plugin_id", pid)
    return pid


def _find_plugin_class(module: Any) -> Optional[Type[Any]]:
    for obj in module.__dict__.values():
        if inspect.isclass(obj) and obj.__name__.endswith(("Plugin", "Routine")):
            return obj
    return None


__all__: List[str] = []

for mod in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
    if mod.name.startswith("_"):
        continue
    module = importlib.import_module(f"{__name__}.{mod.name}")
    __all__.append(mod.name)

    cls = _find_plugin_class(module)
    if cls is None:
        continue
    name = mod.name
    plugin_name = getattr(module, "PLUGIN_NAME", None)
    if name.startswith("wanderer_"):
        base = name[len("wanderer_") :]
        pname = plugin_name or base
        pid = _assign_plugin_id(pname, cls)
        inst = cls()
        inst.plugin_id = pid
        register_wanderer_type(pname, inst)
    elif name.startswith("neuroplasticity_"):
        base = name[len("neuroplasticity_") :]
        pname = plugin_name or base
        pid = _assign_plugin_id(pname, cls)
        inst = cls()
        inst.plugin_id = pid
        register_neuroplasticity_type(pname, inst)
    elif name.startswith("selfattention_"):
        base = name[len("selfattention_") :]
        pname = plugin_name or base
        pid = _assign_plugin_id(pname, cls)
        inst = cls()
        inst.plugin_id = pid
        register_selfattention_type(pname, inst)
    elif name.startswith("synapse_"):
        base = name[len("synapse_") :]
        pname = plugin_name or base
        pid = _assign_plugin_id(pname, cls)
        inst = cls()
        inst.plugin_id = pid
        register_synapse_type(pname, inst)
    elif name.startswith("brain_train_"):
        base = name[len("brain_train_") :]
        pname = plugin_name or base
        pid = _assign_plugin_id(pname, cls)
        inst = cls()
        inst.plugin_id = pid
        register_brain_train_type(pname, inst)
    elif name.startswith("buildingblock_"):
        base = name[len("buildingblock_") :]
        pname = plugin_name or base
        pid = _assign_plugin_id(pname, cls)
        inst = cls()
        inst.plugin_id = pid
        register_buildingblock_type(pname, inst)
    else:
        # Default to neuron plugin registration
        pname = plugin_name or name
        pid = _assign_plugin_id(pname, cls)
        inst = cls()
        inst.plugin_id = pid
        register_neuron_type(pname, inst)

    PLUGIN_GRAPH.add_plugin(pname)
    for dep in getattr(cls, "REQUIRES", []) or []:
        PLUGIN_GRAPH.add_dependency(dep, pname)
    after = getattr(cls, "PHASE_AFTER", None)
    if after:
        PLUGIN_GRAPH.add_dependency(after, pname)


__all__ += ["PLUGIN_ID_REGISTRY"]

