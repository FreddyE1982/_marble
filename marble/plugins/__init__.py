"""Plugin package auto-loader with automatic registration.

All modules inside this package are imported and inspected for plugin
classes. Any class whose name ends with ``Plugin`` or ``Routine`` is
registered automatically based on the module's name. This means dropping a
new module into :mod:`marble.plugins` immediately exposes the plugin without
any manual ``register_*`` calls.
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


def _find_plugin_class(module: Any) -> Optional[Type[Any]]:
    for obj in module.__dict__.values():
        if inspect.isclass(obj) and obj.__name__.endswith(("Plugin", "Routine")):
            return obj
    return None


__all__: List[str] = []

for mod in pkgutil.iter_modules(__path__):
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
        register_wanderer_type(plugin_name or base, cls())
    elif name.startswith("neuroplasticity_"):
        base = name[len("neuroplasticity_") :]
        register_neuroplasticity_type(plugin_name or base, cls())
    elif name.startswith("selfattention_"):
        base = name[len("selfattention_") :]
        register_selfattention_type(plugin_name or base, cls())
    elif name.startswith("synapse_"):
        base = name[len("synapse_") :]
        register_synapse_type(plugin_name or base, cls())
    elif name.startswith("brain_train_"):
        base = name[len("brain_train_") :]
        register_brain_train_type(plugin_name or base, cls())
    else:
        # Default to neuron plugin registration
        register_neuron_type(plugin_name or name, cls())

