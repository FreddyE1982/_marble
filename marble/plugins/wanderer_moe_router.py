"""Mixture-of-Experts router for Wanderer plugin stacks.

The router operates as a meta plugin that observes runtime telemetry for every
registered Wanderer expert (latency, activation counts, niche classification)
and activates a sparse subset per step.  Its gating strategy mirrors
transformer-style MoE routers: softmax probabilities with temperature control,
load-balancing penalties, and a configurable capacity factor that bounds the
number of simultaneously active experts.  Router statistics are mirrored into
the global Reporter tree so the :mod:`marble.decision_controller` can throttle
phase cadence and constraint multipliers based on current utilisation.

All learnable parameters—global router scales and per-expert biases—are exposed
via :func:`~marble.wanderer.expose_learnable_params`, keeping the meta-plugin
fully optimisable alongside the rest of the Wanderer stack.
"""

from __future__ import annotations

import math
import os
import random
import time
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

import torch
import yaml

from ..learnables_yaml import register_learnable_plugin
from ..plugin_telemetry import get_plugin_catalog, get_plugin_usage
from ..reporter import report
from ..wanderer import expose_learnable_params
from ..plugins import PLUGIN_ID_REGISTRY

if TYPE_CHECKING:  # pragma: no cover - hints only
    from ..wanderer import Wanderer
    from ..graph import Neuron, Synapse


def _load_moe_config() -> Dict[str, Any]:
    """Load router configuration from ``config.yaml``.

    Configuration lives under ``decision_controller.moe_routing`` so the router
    and the decision controller share a single source of truth.
    """

    base = os.path.dirname(os.path.dirname(__file__))
    cfg_path = os.path.join(base, "config.yaml")
    try:
        with open(cfg_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception:
        return {}
    dc = data.get("decision_controller", {}) if isinstance(data, dict) else {}
    moe = dc.get("moe_routing", {}) if isinstance(dc, dict) else {}
    return moe if isinstance(moe, dict) else {}


@dataclass
class _Expert:
    """Wrapper that consults :class:`MoERouterPlugin` before delegating."""

    plugin: Any
    name: str
    router: "MoERouterPlugin"

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - passthrough
        return getattr(self.plugin, item)

    # The wrapper methods mirror the Wanderer plugin surface.  We only gate
    # publicly documented hooks; attribute passthrough keeps custom hooks intact.
    def on_init(self, wanderer: "Wanderer") -> None:
        if hasattr(self.plugin, "on_init"):
            self.plugin.on_init(wanderer)

    def before_walk(self, wanderer: "Wanderer", start: "Neuron") -> None:
        if (
            hasattr(self.plugin, "before_walk")
            and self.router.is_active(self.name)
        ):
            self.plugin.before_walk(wanderer, start)

    def after_walk(self, wanderer: "Wanderer", summary: Dict[str, Any]) -> None:
        if hasattr(self.plugin, "after_walk"):
            if self.router.is_active(self.name):
                self.plugin.after_walk(wanderer, summary)

    def on_step(
        self,
        wanderer: "Wanderer",
        current: "Neuron",
        synapse: Optional["Synapse"],
        direction: str,
        step_index: int,
        output: Any,
    ) -> Any:
        if hasattr(self.plugin, "on_step") and self.router.is_active(self.name):
            return self.plugin.on_step(
                wanderer, current, synapse, direction, step_index, output
            )
        return None

    def choose_next(
        self,
        wanderer: "Wanderer",
        current: "Neuron",
        choices: List[Tuple["Synapse", str]],
    ) -> Tuple[Optional["Synapse"], str]:
        if hasattr(self.plugin, "choose_next") and self.router.is_active(self.name):
            return self.plugin.choose_next(wanderer, current, choices)
        return None, "forward"

    def loss(self, wanderer: "Wanderer", outputs: List[Any]) -> Any:
        if hasattr(self.plugin, "loss") and self.router.is_active(self.name):
            return self.plugin.loss(wanderer, outputs)
        torch_mod = getattr(wanderer, "_torch", None)
        if torch_mod is None:
            return 0.0
        return torch_mod.tensor(0.0, device=getattr(wanderer, "_device", "cpu"))


class MoERouterPlugin:
    """Meta-plugin that performs sparse expert routing across Wanderer plugins."""

    def __init__(self) -> None:
        cfg = _load_moe_config()
        self._enabled = bool(cfg.get("enabled", False))
        self._decision_interval = max(1, int(cfg.get("decision_interval", 1)))
        self._capacity_factor = float(cfg.get("capacity_factor", 1.25))
        self._min_active = max(1, int(cfg.get("min_active_experts", 1)))
        self._max_active = max(self._min_active, int(cfg.get("max_active_experts", 4)))
        self._load_balance_alpha = float(cfg.get("load_balance_alpha", 0.15))
        self._load_decay = min(0.999, max(0.0, float(cfg.get("load_balance_decay", 0.8))))
        self._budget_weight = float(cfg.get("budget_weight", 0.25))
        self._rng = random.Random(cfg.get("seed", None))
        self._experts: Dict[str, _Expert] = {}
        self._active: set[str] = set()
        self._load_tracker: Dict[str, float] = defaultdict(float)
        self._last_logits: Dict[str, float] = {}
        self._last_probs: Dict[str, float] = {}
        self._step_index = 0
        self._latency_budget_ms = 1.0
        self._niche_cache: Dict[str, torch.Tensor] = {}
        self._learnable_cache: Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}
        self._catalog_snapshot: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer: "Wanderer",
        *,
        call_scale: float = 0.1,
        latency_scale: float = -0.05,
        load_penalty: float = 0.8,
        router_temperature: float = 1.0,
        capacity_jitter: float = 0.15,
    ) -> Tuple[torch.Tensor, ...]:
        torch_mod = getattr(wanderer, "_torch", torch)
        def _coerce(value: Any) -> torch.Tensor:
            if isinstance(value, torch_mod.Tensor):
                return value.detach().clone().to(dtype=torch_mod.float32)
            return torch_mod.tensor(float(value), dtype=torch_mod.float32)

        return (
            _coerce(call_scale),
            _coerce(latency_scale),
            _coerce(load_penalty),
            _coerce(router_temperature),
            _coerce(capacity_jitter),
        )

    # ------------------------------------------------------------------
    def _ensure_plugin_learnables(
        self, wanderer: "Wanderer", name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        wid = id(wanderer)
        cache = self._learnable_cache.setdefault(wid, {})
        cached = cache.get(name)
        if cached is not None:
            bias, gain = cached
            if bias is not None and gain is not None:
                return bias, gain
        bias_name = f"moe_router_bias_{name}"
        gain_name = f"moe_router_gain_{name}"
        wanderer.ensure_learnable_param(bias_name, 0.0)
        wanderer.ensure_learnable_param(gain_name, 1.0)
        register_learnable_plugin("Wanderer", bias_name, name)
        register_learnable_plugin("Wanderer", gain_name, name)
        bias = wanderer.get_learnable_param_tensor(bias_name)
        gain = wanderer.get_learnable_param_tensor(gain_name)
        cache[name] = (bias, gain)
        return bias, gain

    # ------------------------------------------------------------------
    def _ensure_niche_bias(
        self, wanderer: "Wanderer", niche: Optional[str]
    ) -> torch.Tensor:
        key = niche or "generalist"
        cached = self._niche_cache.get(key)
        if cached is not None:
            return cached
        param_name = f"moe_router_niche_bias_{key}"
        wanderer.ensure_learnable_param(param_name, 0.0)
        register_learnable_plugin("Wanderer", param_name, key)
        tensor = wanderer.get_learnable_param_tensor(param_name)
        self._niche_cache[key] = tensor
        return tensor

    # ------------------------------------------------------------------
    def _normalise_name(self, expert: Any, default: str) -> str:
        explicit = getattr(expert, "PLUGIN_NAME", None)
        if isinstance(explicit, str) and explicit:
            return explicit
        pid = getattr(expert, "plugin_id", None)
        if pid is not None:
            for name, value in PLUGIN_ID_REGISTRY.items():
                if value == pid:
                    return name
        return default

    # ------------------------------------------------------------------
    def _wrap_experts(self, wanderer: "Wanderer") -> None:
        stack: List[Any] = []
        self._experts.clear()
        for plug in list(getattr(wanderer, "_wplugins", []) or []):
            if plug is self:
                stack.append(self)
                continue
            cls_name = plug.__class__.__name__
            canonical = self._normalise_name(plug, cls_name)
            wrapper = _Expert(plug, canonical, self)
            wrapper.plugin_id = getattr(plug, "plugin_id", None)  # type: ignore[attr-defined]
            self._experts[canonical] = wrapper
            stack.append(wrapper)
        wanderer._wplugins = stack
        self._catalog_snapshot = get_plugin_catalog()

    # ------------------------------------------------------------------
    def on_init(self, wanderer: "Wanderer") -> None:
        self._wrap_experts(wanderer)
        if not self._enabled:
            self._active = set(self._experts.keys())

    # ------------------------------------------------------------------
    def before_walk(self, wanderer: "Wanderer", start: "Neuron") -> None:
        self._step_index = 0
        self._refresh_routing(wanderer, reason="walk_start")

    # ------------------------------------------------------------------
    def on_step(
        self,
        wanderer: "Wanderer",
        current: "Neuron",
        synapse: Optional["Synapse"],
        direction: str,
        step_index: int,
        output: Any,
    ) -> None:
        self._step_index += 1
        if self._enabled and self._step_index % self._decision_interval == 0:
            self._refresh_routing(wanderer, reason="step")

    # ------------------------------------------------------------------
    def after_walk(self, wanderer: "Wanderer", summary: Dict[str, Any]) -> None:
        self._refresh_routing(wanderer, reason="walk_end")

    # ------------------------------------------------------------------
    def is_active(self, name: str) -> bool:
        if not self._enabled:
            return True
        if not self._experts:
            return True
        return name in self._active

    # ------------------------------------------------------------------
    def _gather_usage(self, name: str) -> Dict[str, Any]:
        usage = get_plugin_usage()
        if name in usage:
            return usage[name]
        cls_name = self._experts.get(name).plugin.__class__.__name__ if name in self._experts else name
        return usage.get(cls_name, {})

    # ------------------------------------------------------------------
    def _catalog_entry(self, name: str) -> Dict[str, Any]:
        if name in self._catalog_snapshot:
            return self._catalog_snapshot[name]
        entry = get_plugin_catalog().get(name, {})
        if entry:
            self._catalog_snapshot[name] = entry
        return entry

    # ------------------------------------------------------------------
    def _refresh_routing(self, wanderer: "Wanderer", *, reason: str) -> None:
        if not self._experts:
            self._active = set()
            return

        params = self._params(wanderer)
        call_scale = float(params[0].detach().to("cpu").item())
        latency_scale = float(params[1].detach().to("cpu").item())
        load_penalty = float(params[2].detach().to("cpu").item())
        temperature = max(1e-3, float(params[3].detach().to("cpu").item()))
        jitter = float(params[4].detach().to("cpu").item())

        names = list(self._experts.keys())
        logits: List[float] = []
        load_penalties: List[float] = []
        latencies: List[float] = []
        calls: List[int] = []

        for name in names:
            usage = self._gather_usage(name)
            call_count = int(usage.get("calls", 0))
            avg_latency = float(usage.get("avg_latency_ms", 0.0))
            last_latency = float(usage.get("last_latency_ms", avg_latency))
            bias, gain = self._ensure_plugin_learnables(wanderer, name)
            bias_val = float(bias.detach().to("cpu").item())
            gain_val = float(gain.detach().to("cpu").item())
            catalog_entry = self._catalog_entry(name)
            niche = catalog_entry.get("niche") or catalog_entry.get("architecture_role")
            niche_tensor = self._ensure_niche_bias(wanderer, str(niche) if niche else "generalist")
            niche_bias = float(niche_tensor.detach().to("cpu").item())
            load = self._load_tracker.get(name, 0.0)

            score = bias_val
            score += gain_val * math.log1p(max(call_count, 0))
            score += call_scale * math.sqrt(max(call_count, 0))
            score += latency_scale * avg_latency
            score -= load_penalty * load
            score += niche_bias
            score += self._rng.uniform(-jitter, jitter)

            logits.append(score)
            load_penalties.append(load)
            latencies.append(last_latency)
            calls.append(call_count)

        tensor_logits = torch.tensor(logits, dtype=torch.float32)
        if load_penalties and self._load_balance_alpha:
            load_tensor = torch.tensor(load_penalties, dtype=torch.float32)
            tensor_logits = tensor_logits - self._load_balance_alpha * load_tensor
        probs = torch.softmax(tensor_logits / temperature, dim=0)
        total_experts = len(names)
        if total_experts == 0:
            self._active = set()
            return

        base_target = self._capacity_factor * max(1.0, self._latency_budget_ms)
        est_latency = float((probs * torch.tensor(latencies)).sum().item())
        self._latency_budget_ms = (
            (1.0 - self._budget_weight) * self._latency_budget_ms
            + self._budget_weight * max(est_latency, 1e-3)
        )
        capacity = int(round(min(total_experts, max(self._min_active, self._capacity_factor * total_experts))))
        capacity = max(self._min_active, min(self._max_active, capacity))
        topk = capacity
        if topk <= 0:
            topk = 1
        topk = min(topk, total_experts)
        indices = torch.topk(probs, topk).indices.tolist()
        active = {names[i] for i in indices}
        if not active:
            active = {names[int(torch.argmax(probs))]}
        self._active = active

        for name in names:
            prev = self._load_tracker.get(name, 0.0)
            target = 1.0 if name in active else 0.0
            self._load_tracker[name] = self._load_decay * prev + (1.0 - self._load_decay) * target
        self._last_logits = {name: float(tensor_logits[i].item()) for i, name in enumerate(names)}
        self._last_probs = {name: float(probs[i].item()) for i, name in enumerate(names)}

        self._emit_metrics(names, calls, latencies, probs, reason)

    # ------------------------------------------------------------------
    def _emit_metrics(
        self,
        names: Iterable[str],
        calls: List[int],
        latencies: List[float],
        probs: torch.Tensor,
        reason: str,
    ) -> None:
        active_count = len(self._active)
        names_list = list(names)
        loads = [self._load_tracker.get(name, 0.0) for name in names_list]
        if loads:
            avg_load = sum(loads) / len(loads)
            load_balance = math.sqrt(sum((x - avg_load) ** 2 for x in loads) / len(loads))
        else:
            load_balance = 0.0
        expected_latency = float((probs * torch.tensor(latencies)).sum().item())
        budget_pressure = expected_latency / max(self._latency_budget_ms, 1e-3)

        per_plugin: Dict[str, Dict[str, float]] = {}
        for idx, name in enumerate(names_list):
            per_plugin[name] = {
                "calls": float(calls[idx]),
                "probability": float(probs[idx].item()),
                "logit": self._last_logits.get(name, 0.0),
                "load": float(self._load_tracker.get(name, 0.0)),
                "latency_ms": float(latencies[idx]),
                "active": 1.0 if name in self._active else 0.0,
            }

        for name, payload in per_plugin.items():
            try:
                report("plugins", name, payload, "moe_router", "metrics")
            except Exception:
                logging.getLogger(__name__).exception(
                    "Failed to emit MoE router plugin metrics for %s", name
                )

        summary = {
            "active_experts": float(active_count),
            "available_experts": float(len(names_list)),
            "load_balance": float(load_balance),
            "expected_latency_ms": float(expected_latency),
            "latency_budget_ms": float(self._latency_budget_ms),
            "budget_pressure": float(budget_pressure),
            "timestamp": time.time(),
            "reason": reason,
        }
        for key, value in summary.items():
            try:
                report("decision_controller", key, value, "moe_router", "scalars")
            except Exception:
                logging.getLogger(__name__).exception(
                    "Failed to emit MoE router summary metric %s", key
                )


__all__ = ["MoERouterPlugin"]

PLUGIN_NAME = "moe_router"
