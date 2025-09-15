from __future__ import annotations

from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING
from collections import deque

import torch
import numpy as np

from .codec import TensorLike
from .reporter import report

if TYPE_CHECKING:
    from .selfattention import SelfAttention


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
            if self._torch is not None:
                cuda = getattr(self._torch, "cuda", None)
                if callable(getattr(cuda, "is_available", None)) and cuda.is_available():
                    return "cuda"
        except Exception:
            pass
        return "cpu"

    def _ensure_tensor(self, value: Union[TensorLike, Sequence[float], float, int]) -> TensorLike:
        if self._torch is None:
            if isinstance(value, np.ndarray):
                return value.astype(np.float32)
            if isinstance(value, (list, tuple)):
                return np.asarray(value, dtype=np.float32)
            elif isinstance(value, (int, float)):
                return np.asarray([value], dtype=np.float32)
            else:
                return np.asarray(list(value), dtype=np.float32)  # type: ignore[arg-type]
        else:
            if self._is_torch_tensor(value):
                return value  # type: ignore[return-value]
            if isinstance(value, (list, tuple)):
                return self._torch.tensor(list(value), dtype=self._torch.float32, device=self._device)
            elif isinstance(value, (int, float)):
                return self._torch.tensor([float(value)], dtype=self._torch.float32, device=self._device)
            else:
                return self._torch.tensor(value, dtype=self._torch.float32, device=self._device)

    def _report(self, groupname: str, itemname: str, data: Any, *subgroups: str) -> None:
        if isinstance(data, dict):
            payload = dict(data)
            payload["device"] = self._device
        else:
            payload = {"value": data, "device": self._device}
        report(groupname, itemname, payload, *subgroups)

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
        loss_diff_window: int = 10,
    ) -> None:
        super().__init__()
        self.tensor = tensor
        self.weight: float = float(weight)
        self.bias: float = float(bias)
        self.age: int = int(age)
        self.type_name: Optional[str] = type_name
        self.loss_diff_window: int = int(loss_diff_window)
        self.loss_diffs: Deque[float] = deque(maxlen=self.loss_diff_window)
        self.mean_loss_diff: float = 0.0
        self._plugin_state: Dict[str, Any] = {}
        if 'learnable_params' not in self._plugin_state:
            self._plugin_state['learnable_params'] = {}
        self.incoming: List["Synapse"] = []
        self.outgoing: List["Synapse"] = []

        plugin = _NEURON_TYPES.get(self.type_name) if self.type_name else None
        if plugin is not None and hasattr(plugin, "on_init"):
            plugin.on_init(self)  # type: ignore[attr-defined]
        try:
            self._report("neuron", "create", {"weight": self.weight, "bias": self.bias, "age": self.age, "type": self.type_name}, "events")
        except Exception:
            pass

    @property
    def tensor(self) -> TensorLike:
        return self._tensor

    @tensor.setter
    def tensor(self, value: Union[TensorLike, Sequence[float], float, int]) -> None:
        val = self._ensure_tensor(value)
        try:
            from .plugins.wanderer_resource_allocator import track_tensor as _tt
            with _tt(self, "_tensor"):
                self._tensor = val
        except Exception:
            self._tensor = val

    # Disallow copying to maintain graph immutability during training
    def __copy__(self):
        raise TypeError("Neuron instances are immutable and cannot be copied")

    def __deepcopy__(self, memo):
        raise TypeError("Neuron instances are immutable and cannot be deep-copied")

    def connect_to(self, other: "Neuron", *, direction: str = "uni", age: int = 0, type_name: Optional[str] = None) -> "Synapse":
        s = Synapse(self, other, direction=direction, age=age, type_name=type_name)
        try:
            self._report("neuron", "connect_to", {"direction": direction, "age": age, "type": type_name}, "events")
        except Exception:
            pass
        return s

    def receive(self, value: Union[TensorLike, Sequence[float], float, int]) -> None:
        plugin = _NEURON_TYPES.get(self.type_name) if self.type_name else None
        if plugin is not None and hasattr(plugin, "receive"):
            plugin.receive(self, value)  # type: ignore[attr-defined]
            return
        self.tensor = value
        try:
            self._report("neuron", "receive", {"len": int(self.tensor.numel()) if hasattr(self.tensor, "numel") else (len(self.tensor) if isinstance(self.tensor, list) else 1)}, "events")
        except Exception:
            pass

    def record_loss_diff(self, diff: float) -> None:
        self.loss_diffs.append(float(diff))
        if self.loss_diffs:
            self.mean_loss_diff = sum(self.loss_diffs) / len(self.loss_diffs)
        else:
            self.mean_loss_diff = 0.0

    def forward(self, input_value: Optional[Union[TensorLike, Sequence[float], float, int]] = None) -> TensorLike:
        plugin = _NEURON_TYPES.get(self.type_name) if self.type_name else None
        if plugin is not None and hasattr(plugin, "forward"):
            return plugin.forward(self, input_value)  # type: ignore[attr-defined]

        x = self._ensure_tensor(self.tensor if input_value is None else input_value)
        if self._torch is not None and self._is_torch_tensor(x):
            out = x * self.weight + self.bias
        else:
            xl = np.asarray(x, dtype=np.float32)
            out = xl * self.weight + self.bias
        try:
            if self._torch is not None and self._is_torch_tensor(out):
                out_len = int(out.numel())
            else:
                try:
                    out_len = len(out)
                except Exception:
                    out_len = 1
            def _wb_val(v):
                try:
                    if hasattr(v, "detach"):
                        return float(v.detach().to("cpu").view(-1)[0].item())
                    return float(v)
                except Exception:
                    return None
            self._report("neuron", "forward", {"out_len": out_len, "weight": _wb_val(self.weight), "bias": _wb_val(self.bias)}, "metrics")
        except Exception:
            pass
        return out

    def describe_for_selfattention(self) -> Dict[str, Any]:
        """Return core attributes for SelfAttention consumption."""
        pos = getattr(self, "position", None)
        try:
            weight = float(self.weight.detach().to("cpu").item())
        except Exception:
            weight = float(self.weight)
        return {
            "weight": weight,
            "type_name": self.type_name,
            "position": pos,
        }

    def report_to_selfattention(self, sa: Optional["SelfAttention"] = None) -> Dict[str, Any]:
        """Provide a snapshot of this neuron's state to a SelfAttention instance.

        Parameters
        ----------
        sa:
            Optional SelfAttention object that receives the report via its
            internal `_receive_neuron_report` hook.
        """
        info = self.describe_for_selfattention()
        if sa is not None:
            try:
                getattr(sa, "_receive_neuron_report", lambda *_: None)(self, info)
            except Exception:
                pass
        try:
            self._report("neuron", "selfattention_report", info, "events")
        except Exception:
            pass
        return info

    def step_age(self, delta: int = 1) -> None:
        self.age += int(delta)


class Synapse(_DeviceHelper):
    def __init__(
        self,
        source: Union[Neuron, "Synapse"],
        target: Union[Neuron, "Synapse"],
        *,
        direction: str = "uni",
        age: int = 0,
        type_name: Optional[str] = None,
        weight: float = 1.0,
        bias: float = 0.0,
    ) -> None:
        super().__init__()
        if direction not in ("uni", "bi"):
            raise ValueError("direction must be 'uni' or 'bi'")
        self.source: Union[Neuron, Synapse] = source
        self.target: Union[Neuron, Synapse] = target
        self.direction = direction
        self.age = int(age)
        self.type_name: Optional[str] = type_name
        self._plugin_state: Dict[str, Any] = {}
        self.weight: float = float(weight)
        self.bias: float = float(bias)

        self.incoming_synapses: List["Synapse"] = []
        self.outgoing_synapses: List["Synapse"] = []

        if isinstance(self.source, Neuron):
            self.source.outgoing.append(self)
            if self.direction == "bi":
                self.source.incoming.append(self)
        else:
            self.source.outgoing_synapses.append(self)
            if self.direction == "bi":
                self.source.incoming_synapses.append(self)

        if isinstance(self.target, Neuron):
            self.target.incoming.append(self)
            if self.direction == "bi":
                self.target.outgoing.append(self)
        else:
            self.target.incoming_synapses.append(self)
            if self.direction == "bi":
                self.target.outgoing_synapses.append(self)

        plugin = _SYNAPSE_TYPES.get(self.type_name) if self.type_name else None
        if plugin is not None and hasattr(plugin, "on_init"):
            plugin.on_init(self)  # type: ignore[attr-defined]
        try:
            self._report("synapse", "create", {"direction": self.direction, "age": self.age, "weight": self.weight, "bias": self.bias, "type": self.type_name}, "events")
        except Exception:
            pass

    # Disallow copying to maintain graph immutability during training
    def __copy__(self):
        raise TypeError("Synapse instances are immutable and cannot be copied")

    def __deepcopy__(self, memo):
        raise TypeError("Synapse instances are immutable and cannot be deep-copied")

    def transmit(
        self,
        value: Union[TensorLike, Sequence[float], float, int],
        *,
        direction: str = "forward",
        visited: Optional[Dict["Synapse", "Neuron"]] = None,
    ) -> Neuron:
        plugin = _SYNAPSE_TYPES.get(self.type_name) if self.type_name else None
        if plugin is not None and hasattr(plugin, "transmit"):
            return plugin.transmit(self, value, direction=direction)  # type: ignore[attr-defined]

        if direction not in ("forward", "backward"):
            raise ValueError("direction must be 'forward' or 'backward'")

        if visited is None:
            visited = {}
        if self in visited:
            return visited[self]

        val = self._ensure_tensor(value)
        holder = {"val": val}
        applied = False
        try:
            from .plugins.wanderer_resource_allocator import track_tensor as _tt
            with _tt(holder, "val"):
                if self._torch is not None and self._is_torch_tensor(holder["val"]):
                    out = holder["val"] * float(self.weight)
                    out.add_(float(self.bias))
                    holder["val"] = out
                else:
                    vl = np.array(holder["val"], dtype=np.float32, copy=True)
                    vl *= float(self.weight)
                    vl += float(self.bias)
                    holder["val"] = vl
                applied = True
        except Exception:
            if not applied:
                if self._torch is not None and self._is_torch_tensor(holder["val"]):
                    out = holder["val"] * float(self.weight)
                    out.add_(float(self.bias))
                    holder["val"] = out
                else:
                    vl = np.array(holder["val"], dtype=np.float32, copy=True)
                    vl *= float(self.weight)
                    vl += float(self.bias)
                    holder["val"] = vl
        val = holder["val"]

        if direction == "forward":
            if self.direction not in ("uni", "bi"):
                raise ValueError("This synapse does not allow forward transmission")
            dest = self.target
            if isinstance(dest, Synapse):
                visited[self] = None
                out_neuron = dest.transmit(val, direction="forward", visited=visited)
            else:
                dest.receive(val)
                out_neuron = dest
        else:
            if self.direction != "bi":
                raise ValueError("This synapse does not allow backward transmission")
            dest = self.source
            if isinstance(dest, Synapse):
                visited[self] = None
                out_neuron = dest.transmit(val, direction="backward", visited=visited)
            else:
                dest.receive(val)
                out_neuron = dest
        visited[self] = out_neuron
        try:
            self._report(
                "synapse",
                "transmit",
                {"dir": direction, "weight": float(self.weight), "bias": float(self.bias)},
                "events",
            )
        except Exception:
            pass
        return out_neuron

    def connect_to_synapse(self, other: "Synapse", *, direction: str = "forward") -> None:
        if direction == "forward":
            if isinstance(self.target, Neuron):
                try:
                    self.target.incoming.remove(self)
                except Exception:
                    pass
            self.target = other
            self.outgoing_synapses.append(other)
            other.incoming_synapses.append(self)
        else:
            if isinstance(self.source, Neuron):
                try:
                    self.source.outgoing.remove(self)
                except Exception:
                    pass
            self.source = other
            self.incoming_synapses.append(other)
            other.outgoing_synapses.append(self)

    def step_age(self, delta: int = 1) -> None:
        self.age += int(delta)
        try:
            self._report("synapse", "aged", {"age": self.age}, "metrics")
        except Exception:
            pass


__all__ = [
    "_DeviceHelper",
    "register_neuron_type",
    "register_synapse_type",
    "Neuron",
    "Synapse",
]
