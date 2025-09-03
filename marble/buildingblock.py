from __future__ import annotations

"""Registry and base class for BuildingBlock plugins.

BuildingBlocks are tiny plugins that perform atomic graph manipulations.
They can be combined freely to assemble higher level dynamic plugins.
"""

from typing import Any, Dict, Sequence, Optional

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

    def _resolve_device(self, brain: "Brain", device: Optional[str] = None) -> str:
        """Return the desired device, inferring from the brain when missing."""
        return device or getattr(brain, "_device", "cpu")

    def _to_index(self, brain: "Brain", index: Any, *, device: Optional[str] = None) -> tuple:
        """Convert a learnable index tensor into a usable tuple."""
        dev = self._resolve_device(brain, device)
        if hasattr(index, "detach"):
            try:
                # keep on target device unless converting to Python types
                index = index.detach().to(dev)
                index = index.to("cpu").tolist()
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

    def _to_float(
        self,
        value: Any,
        brain: Optional["Brain"] = None,
        *,
        device: Optional[str] = None,
    ) -> Any:
        """Return a float or tensor, respecting the desired device."""
        dev = self._resolve_device(brain, device) if brain is not None else device or "cpu"
        if hasattr(value, "detach"):
            val = value.detach()
            if dev != "cpu":
                return val.to(dev)
            try:
                return float(val.to("cpu").item())
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

