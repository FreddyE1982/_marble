"""Plugin package auto-loader.

Imports every module in this package so plugin classes self-register with
their respective registries. Dropping a new plugin module into this
directory makes it instantly available without manual wiring.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import List

__all__: List[str] = []

for mod in pkgutil.iter_modules(__path__):
    if mod.name.startswith("_"):
        continue
    importlib.import_module(f"{__name__}.{mod.name}")
    __all__.append(mod.name)

