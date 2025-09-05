from __future__ import annotations

"""Wanderer plugin that learns to enable or disable other plugins.

AutoPlugin sits at the top of the Wanderer stack and wraps every other
plugin—including neuroplasticity plugins and building blocks—with a learnable
gate. It respects an optional list of *mandatory* plugins which always remain
active, while every other plugin is free to switch on and off so different
combinations can be explored during training. The gating decision is driven by a
self‑attention score over live training metrics (loss, speed, model complexity)
exposed via :func:`expose_learnable_params` and per-plugin bias/gain terms
registered on the owning :class:`~marble.wanderer.Wanderer` instance. For every
step the gate considers the current walk step and the active neuron.

Learned objective prioritizes overall accuracy, then training speed, followed by
model size and complexity (implicitly via gradients adjusting the gate biases).
Activation and deactivation events are written to an optional CSV log so training
runs can be inspected later.
"""

import math
import os
import time
from typing import Any, Dict, List, Tuple, Optional

import torch

from ..wanderer import expose_learnable_params
from ..buildingblock import get_buildingblock_type, _BUILDINGBLOCK_TYPES


def _load_autoplugin_cfg() -> Dict[str, Any]:
    """Load AutoPlugin configuration from ``config.yaml``."""
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
    return cfg.get("autoplugin", {})


class AutoPlugin:
    """Meta-plugin that toggles other plugins on a per-step basis.

    Parameters
    ----------
    disabled_plugins:
        Optional list of plugin class names that should be fully disabled.
        Any Wanderer or neuroplasticity plugin whose class name appears in
        this list will be removed from the active plugin stacks entirely.
    mandatory_plugins:
        Optional list of plugin class names that must remain active. These
        plugins are still wrapped for logging, but their gates always resolve
        to ``True`` and they can never be disabled.
    log_path:
        Optional path of a CSV log file recording plugin activation events.
    """

    def __init__(
        self,
        disabled_plugins: Optional[List[str]] = None,
        log_path: Optional[str] = None,
        mandatory_plugins: Optional[List[str]] = None,
        decision_interval: Optional[int] = None,
    ) -> None:
        cfg = _load_autoplugin_cfg()
        interval = decision_interval if decision_interval is not None else cfg.get("decision_interval", 1)
        self._decision_interval = max(1, int(interval))
        self._neuro_wrapped = False
        self._sa_wrapped = False
        self._current_neuron_id = 0
        self._disabled = set(disabled_plugins or [])
        self._mandatory = set(mandatory_plugins or [])
        self._log_path = log_path
        if log_path:
            # Create/clear the log file so it's readable immediately during training
            with open(log_path, "w", encoding="utf-8"):
                pass
        self._log_state: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._gate_cache: Dict[Tuple[str, str], Tuple[int, bool]] = {}
        self._theta_mean: Dict[str, torch.Tensor] = {}
        self._theta_cov: Dict[str, torch.Tensor] = {}

    def on_init(self, wanderer: "Wanderer") -> None:  # noqa: D401
        """Wrap existing Wanderer plugins with gating proxies."""

        wplugins = getattr(wanderer, "_wplugins", [])
        new_stack: List[Any] = []
        for p in list(wplugins):
            if p is self:
                new_stack.append(p)
                continue
            name = p.__class__.__name__
            if name in self._disabled:
                continue
            wanderer.ensure_learnable_param(f"autoplugin_bias_{name}", 0.0)
            wanderer.ensure_learnable_param(f"autoplugin_gain_{name}", 1.0)
            new_stack.append(_GatedPlugin(p, name, self))
        wanderer._wplugins = new_stack
        for bb_name in _BUILDINGBLOCK_TYPES.keys():
            wanderer.ensure_learnable_param(f"autoplugin_bias_{bb_name}", 0.0)
            wanderer.ensure_learnable_param(f"autoplugin_gain_{bb_name}", 1.0)

    def before_walk(self, wanderer: "Wanderer", start: "Neuron") -> None:
        """Ensure all plugin stacks are wrapped before training."""

        if not self._neuro_wrapped:
            nplugins = getattr(wanderer, "_neuro_plugins", [])
            new_stack: List[Any] = []
            for p in list(nplugins):
                name = p.__class__.__name__
                if name in self._disabled:
                    continue
                wanderer.ensure_learnable_param(f"autoplugin_bias_{name}", 0.0)
                wanderer.ensure_learnable_param(f"autoplugin_gain_{name}", 1.0)
                new_stack.append(_GatedPlugin(p, name, self))
            wanderer._neuro_plugins = new_stack
            self._neuro_wrapped = True

        if not self._sa_wrapped:
            for sa in getattr(wanderer, "_selfattentions", []) or []:
                routines = getattr(sa, "_routines", [])
                new_routines: List[Any] = []
                for r in list(routines):
                    if isinstance(r, _GatedSARoutine):
                        new_routines.append(r)
                        continue
                    name = r.__class__.__name__
                    if name in self._disabled:
                        continue
                    wanderer.ensure_learnable_param(f"autoplugin_bias_{name}", 0.0)
                    wanderer.ensure_learnable_param(f"autoplugin_gain_{name}", 1.0)
                    new_routines.append(_GatedSARoutine(r, name, self))
                sa._routines = new_routines
            self._sa_wrapped = True

    def _write_log(
        self,
        action: str,
        plugintype: str,
        name: str,
        last_time: float,
        last_steps: int,
        total_steps: int,
    ) -> None:
        if not self._log_path:
            return
        with open(self._log_path, "a", encoding="utf-8") as fp:
            fp.write(
                f"{action},{plugintype},{name},{last_time:.3f},{last_steps},{total_steps}\n"
            )
            fp.flush()
            os.fsync(fp.fileno())

    def _update_log(
        self,
        wanderer: "Wanderer",
        plugintype: str,
        name: str,
        active: bool,
    ) -> None:
        if not self._log_path or name in self._mandatory:
            return
        key = (plugintype, name)
        state = self._log_state.setdefault(
            key,
            {
                "active": False,
                "active_since_step": 0,
                "active_since_time": 0.0,
                "last_active_steps": 0,
                "last_active_time": 0.0,
                "total_active_steps": 0,
                "initialized": False,
            },
        )
        step = getattr(wanderer, "neuron_fire_count", 0)
        now = time.time()
        if not state["initialized"]:
            self._write_log(
                "activated" if active else "deactivated",
                plugintype,
                name,
                0.0,
                0,
                0,
            )
            state["initialized"] = True
            state["active"] = active
            if active:
                state["active_since_step"] = step
                state["active_since_time"] = now
            return
        if active and not state["active"]:
            self._write_log(
                "activated",
                plugintype,
                name,
                state["last_active_time"],
                state["last_active_steps"],
                state["total_active_steps"],
            )
            state["active"] = True
            state["active_since_step"] = step
            state["active_since_time"] = now
        elif not active and state["active"]:
            last_steps = step - state["active_since_step"]
            last_time = now - state["active_since_time"]
            state["last_active_steps"] = last_steps
            state["last_active_time"] = last_time
            state["total_active_steps"] += last_steps
            self._write_log(
                "deactivated",
                plugintype,
                name,
                last_time,
                last_steps,
            state["total_active_steps"],
            )
            state["active"] = False

    def _feature_vector(self, wanderer: "Wanderer", name: str) -> torch.Tensor:
        """Return feature vector ``Φ(h_t, a_t)`` for plugin *name*."""

        loss = float(getattr(wanderer, "_last_walk_mean_loss", 0.0) or 0.0)
        steps = float(getattr(wanderer, "_walk_step_count", 0.0))
        brain = getattr(wanderer, "brain", None)
        complexity = float(
            len(getattr(brain, "neurons", {})) + len(getattr(brain, "synapses", []))
            if brain is not None
            else 0.0
        )
        plugin_id = float(sum(ord(c) for c in name) % 1000) / 1000.0
        return torch.tensor([loss, steps, complexity, plugin_id], dtype=torch.float32)

    def is_active(
        self,
        wanderer: "Wanderer",
        name: str,
        neuron: "Neuron | None",
        plugintype: str = "wanderer",
    ) -> bool:
        """Return True if the named plugin should be active."""

        self._current_neuron_id = 0 if neuron is None else id(neuron)
        if name in self._mandatory:
            self._update_log(wanderer, plugintype, name, True)
            self._gate_cache[(plugintype, name)] = (getattr(wanderer, "neuron_fire_count", 0), True)
            return True
        if name in self._disabled:
            self._update_log(wanderer, plugintype, name, False)
            self._gate_cache[(plugintype, name)] = (getattr(wanderer, "neuron_fire_count", 0), False)
            return False

        step = getattr(wanderer, "neuron_fire_count", 0)
        key = (plugintype, name)
        cached = self._gate_cache.get(key)
        if cached is not None and step % self._decision_interval != 0:
            self._update_log(wanderer, plugintype, name, cached[1])
            return cached[1]

        bias = wanderer.get_learnable_param_tensor(f"autoplugin_bias_{name}")
        wanderer.ensure_learnable_param(f"autoplugin_gain_{name}", 1.0)
        gain = wanderer.get_learnable_param_tensor(f"autoplugin_gain_{name}")
        device = getattr(wanderer, "_device", torch.device("cpu"))
        features = self._feature_vector(wanderer, name).to(device)
        mean = self._theta_mean.setdefault(name, torch.zeros_like(features)).to(device)
        self._theta_mean[name] = mean
        cov = self._theta_cov.setdefault(
            name, torch.eye(features.shape[0], device=device)
        ).to(device)
        self._theta_cov[name] = cov
        theta = torch.distributions.MultivariateNormal(mean, cov).sample().to(device)
        score = (features @ theta) * gain + bias
        gate = torch.sigmoid(score)
        result = bool(gate.detach().to("cpu").item() >= 0.5)
        phi = features.view(-1, 1)
        cov_inv = torch.inverse(cov)
        new_cov = torch.inverse(cov_inv + phi @ phi.T)
        y = torch.tensor(float(result))
        new_mean = new_cov @ (cov_inv @ mean.view(-1, 1) + phi * y)
        self._theta_mean[name] = new_mean.view(-1)
        self._theta_cov[name] = new_cov
        self._update_log(wanderer, plugintype, name, result)
        self._gate_cache[key] = (step, result)
        return result

    def apply_buildingblock(
        self, wanderer: "Wanderer", name: str, *args: Any, **kwargs: Any
    ) -> Any:
        """Apply a building block if its gate is active."""

        block = get_buildingblock_type(name)
        if block is None or name in self._disabled:
            return None
        wanderer.ensure_learnable_param(f"autoplugin_bias_{name}", 0.0)
        wanderer.ensure_learnable_param(f"autoplugin_gain_{name}", 1.0)
        if not self.is_active(wanderer, name, None, plugintype="buildingblock"):
            return None
        return block.apply(wanderer.brain, *args, **kwargs)

    @expose_learnable_params
    def _attention_score(
        self,
        wanderer: "Wanderer",
        *,
        q_loss: float = 1.0,
        q_speed: float = 0.5,
        q_complexity: float = 0.1,
        k_loss: float = 1.0,
        k_speed: float = 0.5,
        k_complexity: float = 0.1,
        v_loss: float = 1.0,
        v_speed: float = 0.5,
        v_complexity: float = 0.1,
    ):
        """Self-attention score over loss, speed and model size."""

        torch = getattr(wanderer, "_torch", None)
        loss = float(getattr(wanderer, "_last_walk_mean_loss", 0.0) or 0.0)
        steps = float(getattr(wanderer, "_walk_step_count", 0.0))
        brain = getattr(wanderer, "brain", None)
        complexity = float(
            len(getattr(brain, "neurons", {})) + len(getattr(brain, "synapses", []))
            if brain is not None
            else 0.0
        )
        if torch is None:
            scores = [
                q_loss * k_loss * loss,
                q_speed * k_speed * steps,
                q_complexity * k_complexity * complexity,
            ]
            m = max(scores)
            exps = [math.exp(s - m) for s in scores]
            denom = sum(exps) or 1.0
            weights = [e / denom for e in exps]
            vals = [v_loss * loss, v_speed * steps, v_complexity * complexity]
            return sum(w * v for w, v in zip(weights, vals))
        device = getattr(wanderer, "_device", "cpu")
        metrics = torch.tensor([loss, steps, complexity], dtype=torch.float32, device=device)

        def _to_tensor(val: Any) -> Any:
            return val if hasattr(val, "to") else torch.tensor(float(val), dtype=torch.float32, device=device)

        Q = torch.stack([_to_tensor(q_loss), _to_tensor(q_speed), _to_tensor(q_complexity)])
        K = torch.stack([_to_tensor(k_loss), _to_tensor(k_speed), _to_tensor(k_complexity)])
        V = torch.stack([_to_tensor(v_loss), _to_tensor(v_speed), _to_tensor(v_complexity)])
        scores = (metrics * Q) * (metrics * K)
        weights = torch.softmax(scores / math.sqrt(3.0), dim=0)
        return torch.sum(weights * (metrics * V))

    def finalize_logs(self, wanderer: "Wanderer") -> None:
        for (ptype, name), state in list(self._log_state.items()):
            if state.get("active"):
                self._update_log(wanderer, ptype, name, False)
        self._log_state.clear()


class _GatedPlugin:
    """Proxy that consults :class:`AutoPlugin` before delegating."""

    def __init__(self, plugin: Any, name: str, controller: AutoPlugin) -> None:
        self._plugin = plugin
        self._name = name
        self._controller = controller

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - passthrough
        return getattr(self._plugin, item)

    def on_init(self, wanderer: "Wanderer") -> None:
        if hasattr(self._plugin, "on_init"):
            self._plugin.on_init(wanderer)

    def before_walk(self, wanderer: "Wanderer", start: "Neuron") -> None:
        if hasattr(self._plugin, "before_walk") and self._controller.is_active(
            wanderer, self._name, start, plugintype="wanderer"
        ):
            self._plugin.before_walk(wanderer, start)

    def choose_next(
        self,
        wanderer: "Wanderer",
        current: "Neuron",
        choices: List[Tuple["Synapse", str]],
    ):
        if hasattr(self._plugin, "choose_next") and self._controller.is_active(
            wanderer, self._name, current, plugintype="wanderer"
        ):
            return self._plugin.choose_next(wanderer, current, choices)
        return None

    def loss(self, wanderer: "Wanderer", outputs: List[Any]):
        if hasattr(self._plugin, "loss") and self._controller.is_active(
            wanderer, self._name, None, plugintype="wanderer"
        ):
            return self._plugin.loss(wanderer, outputs)
        torch = getattr(wanderer, "_torch", None)
        return 0.0 if torch is None else torch.tensor(0.0, device=getattr(wanderer, "_device", "cpu"))


class _GatedSARoutine:
    """Proxy for SelfAttention routines controlled by :class:`AutoPlugin`."""

    def __init__(self, routine: Any, name: str, controller: AutoPlugin) -> None:
        self._routine = routine
        self._name = name
        self._controller = controller

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - passthrough
        return getattr(self._routine, item)

    def on_init(self, selfattention: "SelfAttention") -> None:
        if hasattr(self._routine, "on_init"):
            self._routine.on_init(selfattention)

    def after_step(
        self,
        selfattention: "SelfAttention",
        reporter_ro: Any,
        wanderer: "Wanderer",
        step_index: int,
        ctx: Dict[str, Any],
    ):
        if hasattr(self._routine, "after_step") and self._controller.is_active(
            wanderer, self._name, None, plugintype="selfattention"
        ):
            return self._routine.after_step(selfattention, reporter_ro, wanderer, step_index, ctx)
        return None


__all__ = ["AutoPlugin"]

