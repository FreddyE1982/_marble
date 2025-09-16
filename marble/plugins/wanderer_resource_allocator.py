from __future__ import annotations

import os
import tempfile
import time
import threading
from typing import List, Tuple, Any, Dict
import math
import weakref
from contextlib import contextmanager, nullcontext

import torch
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

from ..wanderer import expose_learnable_params


class TensorRegistry:
    """Track objects that own tensors.

    Objects are stored via weak references so entries disappear once the
    owning object is garbage collected. Only attributes that currently hold
    actual torch tensors are yielded when iterating.
    """

    def __init__(self) -> None:
        self._entries: Dict[Tuple[int, str], weakref.ref] = {}

    def register(self, obj: Any, attr: str) -> None:
        """Register ``obj.attr`` as a tensor to be balanced.

        Duplicate registrations are ignored automatically.
        """

        key = (id(obj), attr)
        if key in self._entries:
            return
        self._entries[key] = weakref.ref(obj, lambda _r, k=key: self._entries.pop(k, None))

    def iter_tensors(self, decay_rate: float):
        """Yield ``(obj, attr, tensor, hits)`` for valid registrations.

        ``hits`` reflects a decayed access count so that long-idle tensors no
        longer dominate the prioritisation. ``decay_rate`` controls the
        exponential decay per second.
        """

        now = time.perf_counter()
        for (oid, attr), ref in list(self._entries.items()):
            obj = ref()
            if obj is None:
                continue
            try:
                t = getattr(obj, attr)
            except AttributeError:
                try:
                    t = obj[attr]  # type: ignore[index]
                except Exception:
                    continue
            except Exception:
                continue
            if torch.is_tensor(t):
                info = getattr(obj, "_tensor_hits", {})
                count, last = info.get(attr, (0.0, now))
                dt = max(0.0, now - last)
                count = count * math.exp(-decay_rate * dt) + 1.0
                info[attr] = (count, now)
                setattr(obj, "_tensor_hits", info)
                yield obj, attr, t, count


TENSOR_REGISTRY = TensorRegistry()
ALLOCATORS: List[weakref.ref] = []


@contextmanager
def track_tensor(obj: Any, attr: str):
    """Register tensors assigned to ``obj.attr`` inside the context.

    Parameters
    ----------
    obj:
        Object or mapping that will receive the tensor.
    attr:
        Attribute name or mapping key to watch.
    """

    yield
    try:
        val = getattr(obj, attr)
    except AttributeError:
        try:
            val = obj[attr]  # type: ignore[index]
        except Exception:
            return
    except Exception:
        return
    if torch.is_tensor(val):
        TENSOR_REGISTRY.register(obj, attr)


def _load_resource_cfg() -> Dict[str, Any]:
    """Load resource allocator configuration from ``config.yaml``.

    The parser is intentionally tiny and only understands the structure used
    in this project::

        resource_allocator:
          max_disk_mb: 20480

    Missing files or malformed entries are treated as empty configuration.
    """

    cfg: Dict[str, Any] = {}
    try:
        base = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(base, "config.yaml"), "r", encoding="utf-8") as fh:
            section = None
            for raw in fh:
                line = raw.split("#", 1)[0].rstrip()
                if not line:
                    continue
                if not line.startswith(" ") and line.endswith(":"):
                    section = line[:-1].strip()
                    cfg[section] = {}
                    continue
                if section and ":" in line:
                    k, v = line.split(":", 1)
                    try:
                        val: Any = float(v.strip())
                    except Exception:
                        val = v.strip()
                    cfg[section][k.strip()] = val
    except Exception:
        return {}
    return cfg.get("resource_allocator", {})


class ResourceAllocatorPlugin:
    """Adaptive resource allocator for Wanderer.

    Monitors system metrics (CPU, RAM, disk, GPU/VRAM if available) together
    with training statistics such as loss trends and hit frequencies. The
    collected signals drive heuristics that decide **where tensors should live**
    (GPU, CPU or disk) but the plugin deliberately keeps its hands off path
    selection. Whatever other wanderer plugins or default logic pick as the
    next synapse, this allocator merely shuffles the data to keep memory
    balanced.
    """

    def __init__(self) -> None:
        cfg = _load_resource_cfg()
        self.max_disk_mb = float(cfg.get("max_disk_mb", 30720))
        self.compress_offload = bool(cfg.get("compress_offload", True))
        self.min_gpu_tensor_mb = float(cfg.get("min_gpu_tensor_mb", 0.0))
        self.ram_offload_threshold = float(cfg.get("ram_offload_threshold", 0.9))
        self.vram_offload_threshold = float(cfg.get("vram_offload_threshold", 0.9))
        self.disk_usage_threshold = float(cfg.get("disk_usage_threshold", 0.95))
        self._disk_used_mb = 0.0
        self._rebalance_thread: threading.Thread | None = None
        self._rebalance_stop = threading.Event()
        self._transfer_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        ALLOCATORS.append(weakref.ref(self))

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer,
        *,
        gpu_weight: float = 0.1,
        vram_weight: float = 0.1,
        cpu_weight: float = 0.1,
        ram_weight: float = 0.1,
        disk_weight: float = 0.1,
        usage_weight: float = 0.1,
        train_speed_weight: float = 0.1,
        model_speed_weight: float = 0.1,
        hit_freq_weight: float = 0.1,
        loss_weight: float = 0.1,
        loss_speed_weight: float = 0.1,
        loss_accel_weight: float = 0.1,
        reserve_ratio: float = 0.5,
        move_threshold: float = 0.5,
        synapse_move_coef: float = 0.1,
        neuron_move_coef: float = 0.1,
        gpu_bias: float = 0.0,
        vram_bias: float = 0.0,
        cpu_bias: float = 0.0,
        ram_bias: float = 0.0,
        disk_bias: float = 0.0,
        usage_bias: float = 0.0,
        train_speed_bias: float = 0.0,
        model_speed_bias: float = 0.0,
        hit_freq_bias: float = 0.0,
        loss_bias: float = 0.0,
        loss_speed_bias: float = 0.0,
        loss_accel_bias: float = 0.0,
        temperature: float = 1.0,
        decay_rate: float = 0.01,
        smoothing: float = 0.5,
        storage_weight: float = 0.1,
        storage_bias: float = 0.0,
    ) -> Tuple[Any, ...]:
        """Expose a plethora of learnable parameters.

        Returns more than thirty tensors controlling the behaviour of the
        allocator. The decorator wraps raw numbers into tensors registered on
        the wanderer instance.
        """

        return (
            gpu_weight,
            vram_weight,
            cpu_weight,
            ram_weight,
            disk_weight,
            usage_weight,
            train_speed_weight,
            model_speed_weight,
            hit_freq_weight,
            loss_weight,
            loss_speed_weight,
            loss_accel_weight,
            reserve_ratio,
            move_threshold,
            synapse_move_coef,
            neuron_move_coef,
            gpu_bias,
            vram_bias,
            cpu_bias,
            ram_bias,
            disk_bias,
            usage_bias,
            train_speed_bias,
            model_speed_bias,
            hit_freq_bias,
            loss_bias,
            loss_speed_bias,
            loss_accel_bias,
            temperature,
            decay_rate,
            smoothing,
            storage_weight,
            storage_bias,
        )

    def on_init(self, wanderer) -> None:
        st = wanderer._plugin_state
        st.setdefault("resource_hits", {})
        st.setdefault("last_loss", None)
        st.setdefault("last_time", time.perf_counter())
        st.setdefault("base_score", 0.0)
        before = set(getattr(wanderer, "_learnables", {}))
        self._params(wanderer)
        after = set(getattr(wanderer, "_learnables", {}))
        for name in after - before:
            wanderer.set_param_optimization(name, enabled=True)
        try:
            self.start_auto_rebalance(wanderer)
        except Exception:
            pass
        cfg = _load_resource_cfg()
        self.max_disk_mb = float(cfg.get("max_disk_mb", 30720))
        self.vram_offload_threshold = float(cfg.get("vram_offload_threshold", 0.9))
        self._disk_used_mb = 0.0

    # Helper metrics -----------------------------------------------------
    def _system_metrics(self) -> dict:
        gpu_usage = 0.0
        vram_usage = 0.0
        if torch.cuda.is_available():  # pragma: no cover - depends on hardware
            try:
                idx = torch.cuda.current_device()
                stats = torch.cuda.memory_stats(idx)
                vram_usage = float(stats.get("allocated_bytes.all.current", 0))
                total = float(torch.cuda.get_device_properties(idx).total_memory)
                if total > 0:
                    gpu_usage = vram_usage / total
            except Exception:
                pass
        cpu_usage = float(psutil.cpu_percent()) if psutil else 0.0
        ram_usage = float(psutil.virtual_memory().percent) / 100.0 if psutil else 0.0
        disk_usage = float(psutil.disk_usage("/").percent) / 100.0 if psutil else 0.0
        return {
            "gpu": gpu_usage,
            "vram": vram_usage,
            "cpu": cpu_usage,
            "ram": ram_usage,
            "disk": disk_usage,
        }

    def _training_metrics(self, wanderer) -> dict:
        st = wanderer._plugin_state
        now = time.perf_counter()
        dt = now - st.get("last_time", now)
        st["last_time"] = now
        loss_t = wanderer._compute_loss(getattr(wanderer._walk_ctx, "outputs", []))
        loss_v = float(loss_t.detach().to("cpu").item()) if torch.is_tensor(loss_t) else float(loss_t)
        last_loss = st.get("last_loss")
        loss_speed = (loss_v - last_loss) / dt if last_loss is not None and dt > 0 else 0.0
        loss_accel = 0.0
        last_speed = st.get("last_loss_speed")
        if last_speed is not None and dt > 0:
            loss_accel = (loss_speed - last_speed) / dt
        st["last_loss_speed"] = loss_speed
        st["last_loss"] = loss_v
        return {
            "loss": loss_v,
            "loss_speed": loss_speed,
            "loss_accel": loss_accel,
            "train_speed": 1.0 / dt if dt > 0 else 0.0,
        }

    def _hit_freq(self, wanderer, syn) -> float:
        st = wanderer._plugin_state["resource_hits"]
        pos = getattr(syn, "position", None)
        st[pos] = st.get(pos, 0) + 1
        return float(st[pos])

    # Device management -------------------------------------------------
    def _offload_to_disk(self, tensor: torch.Tensor) -> str | None:
        """Persist ``tensor`` to a temporary file and return its path.

        Tensor data is stored using a raw ``float16`` memory-mapped file so
        that reloading can avoid reading the entire buffer into RAM at once.
        Offloading is skipped when it would exceed the configured disk budget.
        """

        arr = tensor.detach().to("cpu", dtype=torch.float16).contiguous()
        size_mb = arr.element_size() * arr.nelement() / (1024 * 1024)
        if self._disk_used_mb + size_mb > self.max_disk_mb:
            return None
        fd, path = tempfile.mkstemp(prefix="marble_offload_", suffix=".bin")
        os.close(fd)
        with open(path, "wb") as fh:
            fh.write(arr.numpy().tobytes())
        self._disk_used_mb += size_mb
        return path

    @staticmethod
    def _finalize_tensor(tensor: torch.Tensor, requires_grad: bool) -> torch.Tensor:
        """Ensure ``tensor`` is a leaf with the requested ``requires_grad`` state."""

        if requires_grad:
            if not tensor.is_leaf or not tensor.requires_grad:
                tensor = tensor.detach()
                tensor.requires_grad_(True)
            else:
                tensor.requires_grad_(True)
            return tensor
        if tensor.requires_grad:
            tensor = tensor.detach()
        return tensor

    def _normalise_meta(self, meta: Any, tensor: torch.Tensor, fallback_dtype: torch.dtype) -> Dict[str, Any]:
        """Return a normalised metadata dictionary for an offloaded tensor."""

        if isinstance(meta, dict):
            info = dict(meta)
            info.setdefault("shape", tuple(tensor.shape))
            info.setdefault("dtype", fallback_dtype)
            info.setdefault("storage_dtype", info.get("dtype", tensor.dtype))
            info.setdefault("requires_grad", bool(getattr(tensor, "requires_grad", False)))
            info.setdefault("had_grad_fn", getattr(tensor, "grad_fn", None) is not None)
            return info
        if isinstance(meta, tuple) and len(meta) == 2:
            shape, stored_dtype = meta
            try:
                norm_shape = tuple(int(s) for s in shape)
            except Exception:
                norm_shape = tuple(tensor.shape)
            return {
                "shape": norm_shape,
                "dtype": fallback_dtype,
                "storage_dtype": stored_dtype,
                "requires_grad": bool(getattr(tensor, "requires_grad", False)),
                "had_grad_fn": getattr(tensor, "grad_fn", None) is not None,
                "storage": "disk",
            }
        return {
            "shape": tuple(tensor.shape),
            "dtype": fallback_dtype,
            "storage_dtype": fallback_dtype,
            "requires_grad": bool(getattr(tensor, "requires_grad", False)),
            "had_grad_fn": getattr(tensor, "grad_fn", None) is not None,
        }

    def _restore_from_disk(self, path: str, meta: Dict[str, Any], device: str) -> torch.Tensor:
        """Load tensor data from ``path`` and move it to ``device``."""

        shape = tuple(meta.get("shape", ()))
        if not shape:
            raise RuntimeError("offload metadata is missing tensor shape")
        numel = int(torch.tensor(shape).prod().item())
        mapped = torch.from_file(path, size=numel, dtype=torch.float16).view(*shape)
        storage_dtype = meta.get("storage_dtype", torch.float16)
        if storage_dtype != torch.float16:
            mapped = mapped.to(storage_dtype)
        try:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            self._disk_used_mb = max(0.0, self._disk_used_mb - size_mb)
        except Exception:
            pass
        try:
            os.remove(path)
        except Exception:
            pass
        if device == "cuda" and torch.cuda.is_available():
            stream = self._transfer_stream
            with torch.cuda.stream(stream) if stream else nullcontext():
                mapped = mapped.to(device, non_blocking=True)
        elif device != "disk":
            mapped = mapped.to(device)
        return mapped

    def _safe_transfer(self, obj: Any, attr: str, tensor: torch.Tensor, device: str) -> None:
        """Move ``tensor`` to ``device`` handling VRAM/RAM pressure."""

        if not torch.is_tensor(tensor):
            return

        off_attr = f"_{attr}_offload"
        meta_attr = f"_{attr}_offmeta"
        dtype_attr = f"_{attr}_origdtype"
        off_path = getattr(obj, off_attr, None)
        fallback_dtype = getattr(obj, dtype_attr, tensor.dtype)
        meta_dict = self._normalise_meta(getattr(obj, meta_attr, None), tensor, fallback_dtype)
        needs_grad = bool(getattr(tensor, "requires_grad", False) or getattr(tensor, "grad_fn", None))
        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)
        orig_device = str(tensor.device)

        if isinstance(off_path, str):
            try:
                restored = self._restore_from_disk(off_path, meta_dict, device)
                target_dtype = meta_dict.get("dtype", restored.dtype)
                if target_dtype is not None and restored.dtype != target_dtype:
                    restored = restored.to(target_dtype)
                req_flag = bool(meta_dict.get("requires_grad", False) or meta_dict.get("had_grad_fn", False))
                restored = self._finalize_tensor(restored, req_flag)
                setattr(obj, attr, restored)
                setattr(obj, off_attr, None)
                setattr(obj, meta_attr, None)
                setattr(obj, dtype_attr, None)
            except Exception:
                return
            return

        try:
            if device == "disk":
                cpu_dtype = torch.float16 if self.compress_offload and orig_dtype == torch.float32 else orig_dtype
                cpu_tensor = tensor.to("cpu", dtype=cpu_dtype) if needs_grad else tensor.detach().to("cpu", dtype=cpu_dtype)
                path = self._offload_to_disk(cpu_tensor)
                if path is not None:
                    setattr(obj, off_attr, path)
                    meta = {
                        "shape": orig_shape,
                        "dtype": orig_dtype,
                        "storage_dtype": cpu_tensor.dtype,
                        "requires_grad": bool(getattr(tensor, "requires_grad", False)),
                        "had_grad_fn": getattr(tensor, "grad_fn", None) is not None,
                        "storage": "disk",
                        "last_device": orig_device,
                    }
                    setattr(obj, meta_attr, meta)
                    req_meta = bool(meta["requires_grad"] or meta["had_grad_fn"])
                    cpu_tensor = self._finalize_tensor(cpu_tensor, req_meta)
                    setattr(obj, attr, cpu_tensor)
                    if cpu_dtype != orig_dtype:
                        setattr(obj, dtype_attr, orig_dtype)
                return
            if device == "cuda" and torch.cuda.is_available():
                stream = self._transfer_stream
                data = tensor
                with torch.cuda.stream(stream) if stream else nullcontext():
                    data = data.to(device, non_blocking=True)
                orig_dtype_override = getattr(obj, dtype_attr, None)
                if orig_dtype_override is not None:
                    data = data.to(orig_dtype_override)
                    setattr(obj, dtype_attr, None)
                data = self._finalize_tensor(data, needs_grad)
                setattr(obj, attr, data)
                setattr(obj, meta_attr, {
                    "shape": tuple(data.shape),
                    "dtype": data.dtype,
                    "storage_dtype": data.dtype,
                    "requires_grad": bool(getattr(data, "requires_grad", False)),
                    "had_grad_fn": getattr(data, "grad_fn", None) is not None,
                    "storage": "cuda",
                    "last_device": "cuda",
                })
                setattr(obj, off_attr, None)
            else:
                dtype = (
                    torch.float16
                    if self.compress_offload and orig_dtype == torch.float32 and device == "cpu"
                    else orig_dtype
                )
                moved = tensor.to(device, dtype=dtype) if needs_grad else tensor.detach().to(device, dtype=dtype)
                if dtype != orig_dtype:
                    setattr(obj, dtype_attr, orig_dtype)
                else:
                    setattr(obj, dtype_attr, None)
                moved = self._finalize_tensor(moved, needs_grad)
                setattr(obj, attr, moved)
                setattr(obj, meta_attr, {
                    "shape": tuple(moved.shape),
                    "dtype": orig_dtype,
                    "storage_dtype": moved.dtype,
                    "requires_grad": bool(getattr(tensor, "requires_grad", False)),
                    "had_grad_fn": getattr(tensor, "grad_fn", None) is not None,
                    "storage": device,
                    "last_device": device,
                })
                setattr(obj, off_attr, None)
            return
        except torch.cuda.OutOfMemoryError:
            pass
        except Exception:
            return

        # CUDA OOM: try CPU first, then disk
        try:
            cpu_dtype = torch.float16 if self.compress_offload and orig_dtype == torch.float32 else orig_dtype
            cpu_tensor = tensor.to("cpu", dtype=cpu_dtype) if needs_grad else tensor.detach().to("cpu", dtype=cpu_dtype)
            if cpu_dtype != orig_dtype:
                setattr(obj, dtype_attr, orig_dtype)
            else:
                setattr(obj, dtype_attr, None)
            avail = psutil.virtual_memory().available if psutil else float("inf")
            needed = cpu_tensor.element_size() * cpu_tensor.nelement()
            if avail < needed:
                path = self._offload_to_disk(cpu_tensor)
                if path is not None:
                    setattr(obj, off_attr, path)
                    meta = {
                        "shape": orig_shape,
                        "dtype": orig_dtype,
                        "storage_dtype": cpu_tensor.dtype,
                        "requires_grad": bool(getattr(tensor, "requires_grad", False)),
                        "had_grad_fn": getattr(tensor, "grad_fn", None) is not None,
                        "storage": "disk",
                        "last_device": orig_device,
                    }
                    setattr(obj, meta_attr, meta)
                    req_meta = bool(meta["requires_grad"] or meta["had_grad_fn"])
                    cpu_tensor = self._finalize_tensor(cpu_tensor, req_meta)
                    setattr(obj, attr, cpu_tensor)
            else:
                cpu_tensor = self._finalize_tensor(cpu_tensor, needs_grad)
                setattr(obj, attr, cpu_tensor)
                setattr(obj, meta_attr, {
                    "shape": tuple(cpu_tensor.shape),
                    "dtype": orig_dtype,
                    "storage_dtype": cpu_tensor.dtype,
                    "requires_grad": bool(getattr(tensor, "requires_grad", False)),
                    "had_grad_fn": getattr(tensor, "grad_fn", None) is not None,
                    "storage": "cpu",
                    "last_device": "cpu",
                })
                setattr(obj, off_attr, None)
            return
        except Exception:
            try:
                fallback = tensor.to("cpu") if needs_grad else tensor.detach().to("cpu")
                fallback = self._finalize_tensor(fallback, needs_grad)
                setattr(obj, attr, fallback)
            except Exception:
                setattr(obj, attr, torch.zeros(orig_shape, dtype=orig_dtype))
            setattr(obj, meta_attr, {
                "shape": orig_shape,
                "dtype": orig_dtype,
                "storage_dtype": orig_dtype,
                "requires_grad": bool(getattr(tensor, "requires_grad", False)),
                "had_grad_fn": getattr(tensor, "grad_fn", None) is not None,
                "storage": "cpu",
                "last_device": "cpu",
            })
            setattr(obj, off_attr, None)
            setattr(obj, dtype_attr, None)

    def restore(self, obj: Any, attr: str, device: torch.device | str):
        """Restore ``obj.attr`` to ``device`` if it was offloaded."""

        tensor = getattr(obj, attr, None)
        if not torch.is_tensor(tensor):
            return None
        off_attr = f"_{attr}_offload"
        meta_attr = f"_{attr}_offmeta"
        dtype_attr = f"_{attr}_origdtype"
        meta = self._normalise_meta(getattr(obj, meta_attr, None), tensor, getattr(obj, dtype_attr, tensor.dtype))
        target = str(device)
        path = getattr(obj, off_attr, None)
        try:
            if isinstance(path, str):
                restored = self._restore_from_disk(path, meta, target)
                setattr(obj, off_attr, None)
            else:
                restored = tensor
                if target.startswith("cuda") and torch.cuda.is_available():
                    stream = self._transfer_stream
                    with torch.cuda.stream(stream) if stream else nullcontext():
                        restored = restored.to(target, non_blocking=True)
                elif target != "disk":
                    restored = restored.to(target)
            dtype = meta.get("dtype", restored.dtype)
            if dtype is not None and restored.dtype != dtype:
                restored = restored.to(dtype)
            req_flag = bool(meta.get("requires_grad", False) or meta.get("had_grad_fn", False))
            restored = self._finalize_tensor(restored, req_flag)
            setattr(obj, attr, restored)
            setattr(obj, meta_attr, None)
            setattr(obj, dtype_attr, None)
            return restored
        except Exception:
            return tensor

    def rebalance_all(self, wanderer) -> None:
        """Evaluate system metrics and rebalance every registered tensor."""
        params = self._params(wanderer)
        to_val = lambda t: float(t.detach().to("cpu").item()) if hasattr(t, "detach") else float(t)
        (
            gpu_w,
            vram_w,
            cpu_w,
            ram_w,
            disk_w,
            usage_w,
            train_speed_w,
            model_speed_w,
            hit_freq_w,
            loss_w,
            loss_speed_w,
            loss_accel_w,
            reserve_ratio,
            move_thr,
            syn_move,
            neu_move,
            gpu_b,
            vram_b,
            cpu_b,
            ram_b,
            disk_b,
            usage_b,
            train_speed_b,
            model_speed_b,
            hit_freq_b,
            loss_b,
            loss_speed_b,
            loss_accel_b,
            temp,
            decay,
            smooth,
            storage_w,
            storage_b,
        ) = map(to_val, params)

        sysm = self._system_metrics()
        trainm = self._training_metrics(wanderer)
        base_score = (
            gpu_w * (sysm["gpu"] + gpu_b)
            + vram_w * (sysm["vram"] + vram_b)
            + cpu_w * (sysm["cpu"] + cpu_b)
            + ram_w * (sysm["ram"] + ram_b)
            + disk_w * (sysm["disk"] + disk_b)
            + train_speed_w * (trainm["train_speed"] + train_speed_b)
            + loss_w * (trainm["loss"] + loss_b)
            + loss_speed_w * (trainm["loss_speed"] + loss_speed_b)
            + loss_accel_w * (trainm["loss_accel"] + loss_accel_b)
        )

        st = wanderer._plugin_state
        prev = st.get("base_score", base_score)
        base_score = smooth * prev + (1.0 - smooth) * base_score
        st["base_score"] = base_score
        for obj, attr, t, _hits in TENSOR_REGISTRY.iter_tensors(decay):
            if torch.cuda.is_available() and sysm["gpu"] < self.vram_offload_threshold:
                target_device = "cuda"
            elif (
                sysm["ram"] > self.ram_offload_threshold
                and self._disk_used_mb < self.max_disk_mb
                and sysm["disk"] < self.disk_usage_threshold
            ):
                target_device = "disk"
            else:
                target_device = "cpu"
            self._safe_transfer(obj, attr, t, target_device)

    def start_auto_rebalance(self, wanderer, interval: float = 5.0) -> None:
        if self._rebalance_thread and self._rebalance_thread.is_alive():
            return
        self._rebalance_stop.clear()
        wref = weakref.ref(wanderer)

        def _loop() -> None:
            while not self._rebalance_stop.wait(interval):
                w = wref()
                if w is None:
                    break
                try:
                    self.rebalance_all(w)
                except Exception:
                    pass

        self._rebalance_thread = threading.Thread(target=_loop, daemon=True)
        self._rebalance_thread.start()

    def stop_auto_rebalance(self) -> None:
        self._rebalance_stop.set()
        thr = self._rebalance_thread
        if thr is not None and thr.is_alive():
            thr.join(timeout=0)
        self._rebalance_thread = None

    def clear(self) -> None:
        for obj, attr, _t, _ in list(TENSOR_REGISTRY.iter_tensors(decay_rate=0.0)):
            off_attr = f"_{attr}_offload"
            meta_attr = f"_{attr}_offmeta"
            off_path = getattr(obj, off_attr, None)
            if isinstance(off_path, str):
                try:
                    size_mb = os.path.getsize(off_path) / (1024 * 1024)
                    self._disk_used_mb = max(0.0, self._disk_used_mb - size_mb)
                except Exception:
                    pass
                try:
                    os.remove(off_path)
                except Exception:
                    pass
                setattr(obj, off_attr, None)
                setattr(obj, meta_attr, None)
            try:
                setattr(obj, attr, torch.empty(0))
            except Exception:
                pass
        TENSOR_REGISTRY._entries.clear()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Core logic ---------------------------------------------------------
    def choose_next(self, wanderer, current, choices: List[Tuple["Synapse", str]]):
        """Rebalance tensors without steering the walk."""
        self.rebalance_all(wanderer)
        return None, "forward"


def clear() -> None:
    for ref in list(ALLOCATORS):
        alloc = ref()
        if alloc is not None:
            alloc.clear()


def restore_tensor(obj: Any, attr: str, device: torch.device | str):
    """Restore a tensor tracked by the allocator to ``device``."""

    for ref in list(ALLOCATORS):
        alloc = ref()
        if alloc is None:
            continue
        result = alloc.restore(obj, attr, device)
        if result is not None:
            return result
    tensor = getattr(obj, attr, None)
    if torch.is_tensor(tensor):
        try:
            moved = tensor.to(device)
            if getattr(tensor, "requires_grad", False) and not moved.requires_grad:
                moved.requires_grad_(True)
            setattr(obj, attr, moved)
            return moved
        except Exception:
            return tensor
    return None


__all__ = [
    "ResourceAllocatorPlugin",
    "TensorRegistry",
    "TENSOR_REGISTRY",
    "track_tensor",
    "restore_tensor",
    "clear",
]
PLUGIN_NAME = "resourceallocator"
