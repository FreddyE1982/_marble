from __future__ import annotations

import os
import tempfile
import time
import threading
from typing import List, Tuple, Any, Dict
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

    def iter_tensors(self):
        """Yield ``(obj, attr, tensor, hits)`` for valid registrations.

        Each time a tensor is yielded the access counter for ``obj.attr`` is
        incremented. Callers can use the ``hits`` value to prioritize which
        tensors should stay on fast devices.
        """

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
                hits = info.get(attr, 0) + 1
                info[attr] = hits
                setattr(obj, "_tensor_hits", info)
                yield obj, attr, t, hits


TENSOR_REGISTRY = TensorRegistry()


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
        self.max_disk_mb = float(cfg.get("max_disk_mb", 20480))
        self.compress_offload = bool(cfg.get("compress_offload", True))
        self.min_gpu_tensor_mb = float(cfg.get("min_gpu_tensor_mb", 1.0))
        self.ram_offload_threshold = float(cfg.get("ram_offload_threshold", 0.9))
        self._disk_used_mb = 0.0
        self._rebalance_thread: threading.Thread | None = None
        self._rebalance_stop = threading.Event()
        self._transfer_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

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
        self.max_disk_mb = float(cfg.get("max_disk_mb", 20480))
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

    def _safe_transfer(self, obj: Any, attr: str, tensor: torch.Tensor, device: str) -> None:
        """Move ``tensor`` to ``device`` handling VRAM/RAM pressure.

        On CUDA out-of-memory errors the tensor is first moved to CPU. If RAM
        is also exhausted the tensor is serialized to disk and the attribute is
        cleared while the disk path is stored on the object under
        ``_<attr>_offload`` for later retrieval.
        """

        off_attr = f"_{attr}_offload"
        meta_attr = f"_{attr}_offmeta"
        dtype_attr = f"_{attr}_origdtype"
        off_path = getattr(obj, off_attr, None)
        meta = getattr(obj, meta_attr, None)
        if isinstance(off_path, str) and isinstance(meta, tuple):
            shape, dtype = meta
            try:
                numel = int(torch.tensor(shape).prod().item())
                mapped = torch.from_file(off_path, size=numel, dtype=torch.float16).view(*shape)
                if dtype != torch.float16:
                    mapped = mapped.to(dtype)
                if device == "cuda" and torch.cuda.is_available():
                    stream = self._transfer_stream
                    with torch.cuda.stream(stream) if stream else nullcontext():
                        tensor.data = mapped.to(device, non_blocking=True)
                    if dtype != torch.float16:
                        tensor.data = tensor.data.to(dtype)
                else:
                    tensor.data = mapped.to(device)
                try:
                    size_mb = os.path.getsize(off_path) / (1024 * 1024)
                    self._disk_used_mb = max(0.0, self._disk_used_mb - size_mb)
                except Exception:
                    pass
                os.remove(off_path)
                setattr(obj, off_attr, None)
                setattr(obj, meta_attr, None)
            except Exception:
                return
            return

        try:
            if device == "disk":
                path = self._offload_to_disk(tensor)
                if path is not None:
                    setattr(obj, off_attr, path)
                    setattr(obj, meta_attr, (tensor.shape, getattr(obj, dtype_attr, tensor.dtype)))
                    tensor.data = torch.empty(0)
                return
            if device == "cuda" and torch.cuda.is_available():
                stream = self._transfer_stream
                with torch.cuda.stream(stream) if stream else nullcontext():
                    tensor.data = tensor.data.to(device, non_blocking=True)
                orig_dtype = getattr(obj, dtype_attr, None)
                if orig_dtype is not None:
                    tensor.data = tensor.data.to(orig_dtype)
                    setattr(obj, dtype_attr, None)
            else:
                dtype = (
                    torch.float16
                    if self.compress_offload and tensor.dtype == torch.float32 and device == "cpu"
                    else tensor.dtype
                )
                if dtype != tensor.dtype:
                    setattr(obj, dtype_attr, tensor.dtype)
                tensor.data = tensor.detach().to(device, dtype=dtype)
            return
        except torch.cuda.OutOfMemoryError:
            pass
        except Exception:
            return

        # CUDA OOM: try CPU first, then disk
        try:
            cpu_dtype = torch.float16 if self.compress_offload and tensor.dtype == torch.float32 else tensor.dtype
            if cpu_dtype != tensor.dtype:
                setattr(obj, dtype_attr, tensor.dtype)
            cpu_tensor = tensor.detach().to("cpu", dtype=cpu_dtype)
            avail = psutil.virtual_memory().available if psutil else float("inf")
            needed = cpu_tensor.element_size() * cpu_tensor.nelement()
            if avail < needed:
                path = self._offload_to_disk(cpu_tensor)
                if path is not None:
                    setattr(obj, off_attr, path)
                    setattr(obj, meta_attr, (tensor.shape, getattr(obj, dtype_attr, tensor.dtype)))
                tensor.data = torch.empty(0)
            else:
                tensor.data = cpu_tensor
        except Exception:
            tensor.data = torch.empty(0)

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

        for obj, attr, t, hits in TENSOR_REGISTRY.iter_tensors():
            size_mb = t.element_size() * t.nelement() / (1024 * 1024)
            score = base_score + hit_freq_w * (hits + hit_freq_b) - storage_w * (size_mb + storage_b)
            target_device = "cpu"
            if (
                size_mb > self.min_gpu_tensor_mb
                and torch.cuda.is_available()
                and sysm["gpu"] < reserve_ratio
                and score > move_thr
            ):
                target_device = "cuda"
            elif (
                sysm["ram"] > self.ram_offload_threshold
                and self._disk_used_mb < self.max_disk_mb
                and score < -move_thr
            ):
                target_device = "disk"
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

    # Core logic ---------------------------------------------------------
    def choose_next(self, wanderer, current, choices: List[Tuple["Synapse", str]]):
        """Rebalance tensors without steering the walk."""
        self.rebalance_all(wanderer)
        return None, "forward"


__all__ = [
    "ResourceAllocatorPlugin",
    "TensorRegistry",
    "TENSOR_REGISTRY",
    "track_tensor",
]
PLUGIN_NAME = "resourceallocator"
