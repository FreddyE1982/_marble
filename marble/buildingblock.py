from __future__ import annotations

"""Registry and base class for BuildingBlock plugins.

BuildingBlocks are tiny plugins that perform atomic graph manipulations.
They can be combined freely to assemble higher level dynamic plugins.
"""

from typing import Any, Dict, Sequence

from .graph import _DeviceHelper

_BUILDINGBLOCK_TYPES: Dict[str, Any] = {}


def register_buildingblock_type(name: str, plugin: Any) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("BuildingBlock type name must be a non-empty string")
    mod = getattr(getattr(plugin, "__class__", object), "__module__", "")
    if not isinstance(mod, str):
        mod = str(mod)
    if mod.startswith("marble.") and not mod.startswith("marble.plugins."):
        raise ValueError(
            f"BuildingBlock plugin '{name}' must live under marble.plugins.*; got module '{mod}'"
        )
    _BUILDINGBLOCK_TYPES[name] = plugin


class BuildingBlock(_DeviceHelper):
    """Base class for all BuildingBlock plugins."""

    def _to_index(self, brain: "Brain", index: Any) -> tuple:
        """Convert a learnable index tensor into a usable tuple."""
        if hasattr(index, "detach"):
            try:
                index = index.detach().to("cpu").tolist()
            except Exception:
                index = [index.detach().to("cpu").item()]
        if not isinstance(index, Sequence):
            index = [index]
        out = []
        for v in index:
            if hasattr(v, "detach"):
                v = v.detach().to("cpu").item()
            out.append(int(float(v)))
        if getattr(brain, "mode", "grid") == "grid":
            return tuple(out)
        return tuple(float(v) for v in out)

    def _to_float(self, value: Any) -> float:
        if hasattr(value, "detach"):
            try:
                return float(value.detach().to("cpu").item())
            except Exception:
                pass
        return float(value)

    def apply(self, brain: "Brain", *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - interface
        raise NotImplementedError


def get_buildingblock_type(name: str) -> Any:
    return _BUILDINGBLOCK_TYPES.get(name)


__all__ = [
    "register_buildingblock_type",
    "get_buildingblock_type",
    "BuildingBlock",
]

