from __future__ import annotations

import time
from typing import List, Tuple, Any

import torch
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

from ..wanderer import expose_learnable_params


class ResourceAllocatorPlugin:
    """Adaptive resource allocator for Wanderer paths.

    Tracks system metrics (CPU, RAM, disk, GPU/VRAM if available) along with
    training statistics such as loss trends and hit frequencies. A large set of
    learnable parameters combines these metrics to decide which synapse to
    follow and whether tensors should be moved across devices.
    """

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

    # Core logic ---------------------------------------------------------
    def choose_next(self, wanderer, current, choices: List[Tuple["Synapse", str]]):
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
        gpu_w, vram_w, cpu_w, ram_w, disk_w, usage_w, train_speed_w, model_speed_w, hit_freq_w, loss_w, loss_speed_w, loss_accel_w, reserve_ratio, move_thr, syn_move, neu_move, gpu_b, vram_b, cpu_b, ram_b, disk_b, usage_b, train_speed_b, model_speed_b, hit_freq_b, loss_b, loss_speed_b, loss_accel_b, temp, decay, smooth, storage_w, storage_b = map(to_val, [
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
        ])

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
        scores = []
        for syn, dir_str in choices:
            hits = self._hit_freq(wanderer, syn)
            weight = float(getattr(syn, "weight", 1.0))
            syn_score = base_score + hit_freq_w * (hits + hit_freq_b) + usage_w * (weight + usage_b)
            scores.append((syn_score, syn, dir_str))
        scores.sort(key=lambda x: x[0], reverse=True)
        chosen_score, syn, dir_str = scores[0]
        # Device movement heuristic
        target_device = "cpu"
        if torch.cuda.is_available() and sysm["gpu"] < reserve_ratio:
            if chosen_score > move_thr:
                target_device = "cuda"
        # Move tensor attributes if possible
        for obj in (syn, current):
            for attr, coef in (("weight", syn_move), ("tensor", neu_move)):
                t = getattr(obj, attr, None)
                if torch.is_tensor(t):
                    try:
                        t.data = t.data.to(target_device)
                    except Exception:
                        pass
        return syn, dir_str


__all__ = ["ResourceAllocatorPlugin"]
PLUGIN_NAME = "resourceallocator"
