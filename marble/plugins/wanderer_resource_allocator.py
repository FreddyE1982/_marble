from __future__ import annotations

import os
import tempfile
import time
from typing import List, Tuple, Any, Dict
import weakref

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
        """Yield ``(obj, attr, tensor)`` triples for valid registrations."""

        for (oid, attr), ref in list(self._entries.items()):
            obj = ref()
            if obj is None or not hasattr(obj, attr):
                continue
            t = getattr(obj, attr)
            if torch.is_tensor(t):
                yield obj, attr, t


TENSOR_REGISTRY = TensorRegistry()


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
        self._disk_used_mb = 0.0

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

        Offloading is skipped when it would exceed the configured disk budget.
        """
        size_mb = tensor.element_size() * tensor.nelement() / (1024 * 1024)
        if self._disk_used_mb + size_mb > self.max_disk_mb:
            return None
        fd, path = tempfile.mkstemp(prefix="marble_offload_", suffix=".pt")
        os.close(fd)
        torch.save(tensor, path)
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
        off_path = getattr(obj, off_attr, None)
        if isinstance(off_path, str):
            try:
                tensor.data = torch.load(off_path, map_location=device)
                try:
                    size_mb = os.path.getsize(off_path) / (1024 * 1024)
                    self._disk_used_mb = max(0.0, self._disk_used_mb - size_mb)
                except Exception:
                    pass
                os.remove(off_path)
                setattr(obj, off_attr, None)
            except Exception:
                return
            return

        try:
            tensor.data = tensor.data.to(device)
            return
        except torch.cuda.OutOfMemoryError:
            pass
        except Exception:
            return

        # CUDA OOM: try CPU first, then disk
        try:
            cpu_tensor = tensor.detach().to("cpu")
            avail = psutil.virtual_memory().available if psutil else float("inf")
            needed = cpu_tensor.element_size() * cpu_tensor.nelement()
            if avail < needed:
                path = self._offload_to_disk(cpu_tensor)
                if path is not None:
                    setattr(obj, off_attr, path)
                tensor.data = torch.empty(0)
            else:
                tensor.data = cpu_tensor
        except Exception:
            tensor.data = torch.empty(0)

    # Core logic ---------------------------------------------------------
    def choose_next(self, wanderer, current, choices: List[Tuple["Synapse", str]]):
        """Rebalance tensors without steering the walk.

        The allocator inspects system and training metrics to guess whether
        tensors should migrate between devices. It iterates over the current
        neuron and all candidate synapses, moving their tensors as needed, then
        returns ``None`` so the previously chosen path remains untouched.
        """
        if not choices:
            return None, "forward"
        params = self._params(wanderer)
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
        ) = params
        # detach parameter values
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
        ) = map(
            to_val,
            [
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
            ],
        )

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

        target_device = "cpu"
        if torch.cuda.is_available() and sysm["gpu"] < reserve_ratio:
            if base_score > move_thr:
                target_device = "cuda"

        for obj, attr, t in TENSOR_REGISTRY.iter_tensors():
            self._safe_transfer(obj, attr, t, target_device)

        return None, "forward"


__all__ = ["ResourceAllocatorPlugin", "TensorRegistry", "TENSOR_REGISTRY"]
PLUGIN_NAME = "resourceallocator"
