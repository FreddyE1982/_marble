from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .codec import TensorLike
from .reporter import report


class _DeviceHelper:
    def __init__(self) -> None:
        self._torch = self._try_import_torch()
        self._device = self._select_device()

    def _try_import_torch(self):
        try:
            import torch  # type: ignore
            _ = torch.tensor([0], dtype=torch.long, device="cpu")
            return torch
        except Exception:
            return None

    def _select_device(self) -> str:
        try:
            if self._torch is not None and self._torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _ensure_tensor(self, value: Union[TensorLike, Sequence[float], float, int]) -> TensorLike:
        if self._torch is None:
            if isinstance(value, (list, tuple)):
                return [float(x) for x in value]
            elif isinstance(value, (int, float)):
                return [float(value)]
            else:
                return list(value)  # type: ignore[arg-type]
        else:
            if self._is_torch_tensor(value):
                return value  # type: ignore[return-value]
            if isinstance(value, (list, tuple)):
                return self._torch.tensor(list(value), dtype=self._torch.float32, device=self._device)
            elif isinstance(value, (int, float)):
                return self._torch.tensor([float(value)], dtype=self._torch.float32, device=self._device)
            else:
                return self._torch.tensor(value, dtype=self._torch.float32, device=self._device)

    def _is_torch_tensor(self, obj: Any) -> bool:
        try:
            if self._torch is None:
                return False
            Tensor = self._torch.Tensor  # type: ignore[attr-defined]
            return isinstance(obj, Tensor)
        except Exception:
            return False


_NEURON_TYPES: Dict[str, Any] = {}
_SYNAPSE_TYPES: Dict[str, Any] = {}


def register_neuron_type(name: str, plugin: Any) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("Neuron type name must be a non-empty string")
    mod = getattr(getattr(plugin, "__class__", object), "__module__", "")
    if not isinstance(mod, str):
        mod = str(mod)
    if mod.startswith("marble.") and not mod.startswith("marble.plugins."):
        raise ValueError(f"Neuron plugin '{name}' must be defined in its own module under marble.plugins.*; got module '{mod}'")
    _NEURON_TYPES[name] = plugin


def register_synapse_type(name: str, plugin: Any) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("Synapse type name must be a non-empty string")
    mod = getattr(getattr(plugin, "__class__", object), "__module__", "")
    if not isinstance(mod, str):
        mod = str(mod)
    if mod.startswith("marble.") and not mod.startswith("marble.plugins."):
        raise ValueError(f"Synapse plugin '{name}' must be defined in its own module under marble.plugins.*; got module '{mod}'")
    _SYNAPSE_TYPES[name] = plugin


class Neuron(_DeviceHelper):
    def __init__(
        self,
        tensor: Union[TensorLike, Sequence[float], float, int],
        *,
        weight: float = 1.0,
        bias: float = 0.0,
        age: int = 0,
        type_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.tensor: TensorLike = self._ensure_tensor(tensor)
        self.weight: float = float(weight)
        self.bias: float = float(bias)
        self.age: int = int(age)
        self.type_name: Optional[str] = type_name
        self._plugin_state: Dict[str, Any] = {}
        if 'learnable_params' not in self._plugin_state:
            self._plugin_state['learnable_params'] = {}
        self.incoming: List["Synapse"] = []
        self.outgoing: List["Synapse"] = []

        plugin = _NEURON_TYPES.get(self.type_name) if self.type_name else None
        if plugin is not None and hasattr(plugin, "on_init"):
            plugin.on_init(self)  # type: ignore[attr-defined]
        try:
            report("neuron", "create", {"weight": self.weight, "bias": self.bias, "age": self.age, "type": self.type_name}, "events")
        except Exception:
            pass

    def connect_to(self, other: "Neuron", *, direction: str = "uni", age: int = 0, type_name: Optional[str] = None) -> "Synapse":
        s = Synapse(self, other, direction=direction, age=age, type_name=type_name)
        try:
            report("neuron", "connect_to", {"direction": direction, "age": age, "type": type_name}, "events")
        except Exception:
            pass
        return s

    def receive(self, value: Union[TensorLike, Sequence[float], float, int]) -> None:
        plugin = _NEURON_TYPES.get(self.type_name) if self.type_name else None
        if plugin is not None and hasattr(plugin, "receive"):
            plugin.receive(self, value)  # type: ignore[attr-defined]
            return
        self.tensor = self._ensure_tensor(value)
        try:
            report("neuron", "receive", {"len": int(self.tensor.numel()) if hasattr(self.tensor, "numel") else (len(self.tensor) if isinstance(self.tensor, list) else 1)}, "events")
        except Exception:
            pass

    def forward(self, input_value: Optional[Union[TensorLike, Sequence[float], float, int]] = None) -> TensorLike:
        plugin = _NEURON_TYPES.get(self.type_name) if self.type_name else None
        if plugin is not None and hasattr(plugin, "forward"):
            return plugin.forward(self, input_value)  # type: ignore[attr-defined]

        x = self._ensure_tensor(self.tensor if input_value is None else input_value)
        if self._torch is not None and self._is_torch_tensor(x):
            out = x * self.weight + self.bias
        else:
            xl = x if isinstance(x, list) else list(x)  # type: ignore[arg-type]
            out = [self.weight * float(v) + self.bias for v in xl]
        try:
            out_len = int(out.numel()) if (self._torch is not None and self._is_torch_tensor(out)) else (len(out) if isinstance(out, list) else 1)
            def _wb_val(v):
                try:
                    if hasattr(v, "detach"):
                        return float(v.detach().to("cpu").view(-1)[0].item())
                    return float(v)
                except Exception:
                    return None
            report("neuron", "forward", {"out_len": out_len, "weight": _wb_val(self.weight), "bias": _wb_val(self.bias)}, "metrics")
        except Exception:
            pass
        return out

    def step_age(self, delta: int = 1) -> None:
        self.age += int(delta)


class Synapse(_DeviceHelper):
    def __init__(self, source: Neuron, target: Neuron, *, direction: str = "uni", age: int = 0, type_name: Optional[str] = None, weight: float = 1.0) -> None:
        super().__init__()
        if direction not in ("uni", "bi"):
            raise ValueError("direction must be 'uni' or 'bi'")
        self.source = source
        self.target = target
        self.direction = direction
        self.age = int(age)
        self.type_name: Optional[str] = type_name
        self._plugin_state: Dict[str, Any] = {}
        self.weight: float = float(weight)

        self.source.outgoing.append(self)
        self.target.incoming.append(self)
        if self.direction == "bi":
            self.source.incoming.append(self)
            self.target.outgoing.append(self)

        plugin = _SYNAPSE_TYPES.get(self.type_name) if self.type_name else None
        if plugin is not None and hasattr(plugin, "on_init"):
            plugin.on_init(self)  # type: ignore[attr-defined]
        try:
            report("synapse", "create", {"direction": self.direction, "age": self.age, "weight": self.weight, "type": self.type_name}, "events")
        except Exception:
            pass

    def transmit(self, value: Union[TensorLike, Sequence[float], float, int], *, direction: str = "forward") -> None:
        plugin = _SYNAPSE_TYPES.get(self.type_name) if self.type_name else None
        if plugin is not None and hasattr(plugin, "transmit"):
            plugin.transmit(self, value, direction=direction)  # type: ignore[attr-defined]
            return

        if direction not in ("forward", "backward"):
            raise ValueError("direction must be 'forward' or 'backward'")

        val = self._ensure_tensor(value)
        if self._torch is not None and self._is_torch_tensor(val):
            val = val * float(self.weight)
        else:
            vl = val if isinstance(val, list) else list(val)  # type: ignore[arg-type]
            val = [float(self.weight) * float(v) for v in vl]

        if direction == "forward":
            if self.direction in ("uni", "bi"):
                self.target.receive(val)
            else:
                raise ValueError("This synapse does not allow forward transmission")
        else:
            if self.direction == "bi":
                self.source.receive(val)
            else:
                raise ValueError("This synapse does not allow backward transmission")
        try:
            report("synapse", "transmit", {"dir": direction, "weight": float(self.weight)}, "events")
        except Exception:
            pass

    def step_age(self, delta: int = 1) -> None:
        self.age += int(delta)
        try:
            report("synapse", "aged", {"age": self.age}, "metrics")
        except Exception:
            pass


__all__ = [
    "_DeviceHelper",
    "register_neuron_type",
    "register_synapse_type",
    "Neuron",
    "Synapse",
]
