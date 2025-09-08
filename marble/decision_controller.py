"""Utility to decide plugin actions with constraints.

This controller selects plugin actions while respecting incompatibility sets,
per-plugin capacity limits and a global budget read from ``config.yaml``.

The function :func:`decide_actions` accepts the current plugin hints ``h_t``,
proposed actions ``x_t`` and a ``history`` of previous action sets. It returns a
subset of actions that satisfy all constraints.
"""

from __future__ import annotations

import importlib
import os
import time
import math
from typing import Any, Dict, Iterable, List, Set
from collections import deque

import torch
import torch.nn as nn
import yaml

from .constraints import (
    check_budget,
    check_incompatibility,
    check_linear_constraints,
    check_throughput,
)
from .plugin_graph import PLUGIN_GRAPH
from .plugin_encoder import PluginEncoder
from .action_sampler import select_plugins, compute_logits, sample_actions
from .reward_shaper import RewardShaper
from .offpolicy import Trajectory, doubly_robust
from .policy_gradient import PolicyGradientAgent
from .bayesian_policy import BayesianPolicy
from .plugins import PLUGIN_ID_REGISTRY
from .reporter import REPORTER
from .history_encoder import HistoryEncoder
from . import plugin_cost_profiler as _pcp

# Incompatibility sets I_t: mapping plugin name to set of incompatible plugins
INCOMPATIBILITY_SETS: Dict[str, Set[str]] = {
    "A": {"C"},
    "C": {"A"},
}

# Capacity limits c: maximum times a plugin can appear in history + current step
CAPACITY_LIMITS: Dict[str, int] = {
    "A": 2,
    "B": 1,
    "C": 1,
}


def _load_dwell_bonus() -> float:
    """Load dwell-time cost bonus from ``config.yaml``."""
    base = os.path.dirname(os.path.dirname(__file__))
    cfg: Dict[str, Dict[str, Any]] = {}
    try:
        with open(os.path.join(base, "config.yaml"), "r", encoding="utf-8") as fh:
            section: str | None = None
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
                        cfg[section][k.strip()] = float(v.strip())
                    except Exception:
                        cfg[section][k.strip()] = v.strip()
    except Exception:
        return 0.0
    dc = cfg.get("decision_controller", {})
    try:
        return float(dc.get("dwell_bonus", 0.0))
    except Exception:
        return 0.0


def _load_policy_mode() -> str:
    """Load policy mode from ``config.yaml``."""
    base = os.path.dirname(os.path.dirname(__file__))
    try:
        with open(os.path.join(base, "config.yaml"), "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception:
        return "policy-gradient"
    dc = cfg.get("decision_controller", {})
    return str(dc.get("policy_mode", "policy-gradient"))


def _load_budget() -> float:
    """Load budget limit from ``config.yaml``; default to ``10.0``."""
    base = os.path.dirname(os.path.dirname(__file__))
    cfg: Dict[str, Dict[str, Any]] = {}
    try:
        with open(os.path.join(base, "config.yaml"), "r", encoding="utf-8") as fh:
            section: str | None = None
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
                        cfg[section][k.strip()] = float(v.strip())
                    except Exception:
                        cfg[section][k.strip()] = v.strip()
    except Exception:
        return 10.0
    dc = cfg.get("decision_controller", {})
    try:
        return float(dc.get("budget", 10.0))
    except Exception:
        return 10.0


BUDGET_LIMIT = _load_budget()

# Cache of most recently observed per-plugin costs so that missing
# ``h_t[name]['cost']`` entries can be populated on subsequent calls.
PLUGIN_COST_CACHE: Dict[str, float] = {}


def _load_l1_penalty() -> float:
    """Load L1 penalty for contribution regressor from ``config.yaml``."""
    base = os.path.dirname(os.path.dirname(__file__))
    cfg: Dict[str, Dict[str, Any]] = {}
    try:
        with open(os.path.join(base, "config.yaml"), "r", encoding="utf-8") as fh:
            section: str | None = None
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
                        cfg[section][k.strip()] = float(v.strip())
                    except Exception:
                        cfg[section][k.strip()] = v.strip()
    except Exception:
        return 0.0
    dc = cfg.get("decision_controller", {})
    try:
        return float(dc.get("contribution_l1", 0.0))
    except Exception:
        return 0.0


L1_PENALTY = _load_l1_penalty()


def _load_tau_threshold() -> float:
    """Load minimum state-change interval from ``config.yaml``."""
    base = os.path.dirname(os.path.dirname(__file__))
    cfg: Dict[str, Dict[str, Any]] = {}
    try:
        with open(os.path.join(base, "config.yaml"), "r", encoding="utf-8") as fh:
            section: str | None = None
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
                        cfg[section][k.strip()] = float(v.strip())
                    except Exception:
                        cfg[section][k.strip()] = v.strip()
    except Exception:
        return 0.0
    dc = cfg.get("decision_controller", {})
    try:
        return float(dc.get("tau_threshold", 0.0))
    except Exception:
        return 0.0


TAU_THRESHOLD = _load_tau_threshold()


def _load_lambda_lr() -> float:
    """Load lambda learning rate from ``config.yaml``."""
    base = os.path.dirname(os.path.dirname(__file__))
    cfg: Dict[str, Dict[str, Any]] = {}
    try:
        with open(os.path.join(base, "config.yaml"), "r", encoding="utf-8") as fh:
            section: str | None = None
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
                        cfg[section][k.strip()] = float(v.strip())
                    except Exception:
                        cfg[section][k.strip()] = v.strip()
    except Exception:
        return 0.1
    dc = cfg.get("decision_controller", {})
    try:
        return float(dc.get("lambda_lr", 0.1))
    except Exception:
        return 0.1


LAMBDA_LR = _load_lambda_lr()


def _load_cadence() -> int:
    """Load decision cadence from ``config.yaml``."""
    base = os.path.dirname(os.path.dirname(__file__))
    cfg: Dict[str, Dict[str, Any]] = {}
    try:
        with open(os.path.join(base, "config.yaml"), "r", encoding="utf-8") as fh:
            section: str | None = None
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
                        cfg[section][k.strip()] = float(v.strip())
                    except Exception:
                        cfg[section][k.strip()] = v.strip()
    except Exception:
        return 1
    dc = cfg.get("decision_controller", {})
    try:
        val = int(dc.get("cadence", 1))
        return max(1, val)
    except Exception:
        return 1


CADENCE = _load_cadence()
STEP_COUNTER = 0

DWELL_BONUS = _load_dwell_bonus()
POLICY_MODE = _load_policy_mode()
DWELL_COUNT: Dict[str, int] = {}


def _load_dwell_threshold() -> float:
    """Load dwell-time step threshold ``D`` from ``config.yaml``."""
    base = os.path.dirname(os.path.dirname(__file__))
    cfg: Dict[str, Dict[str, Any]] = {}
    try:
        with open(os.path.join(base, "config.yaml"), "r", encoding="utf-8") as fh:
            section: str | None = None
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
                        cfg[section][k.strip()] = float(v.strip())
                    except Exception:
                        cfg[section][k.strip()] = v.strip()
    except Exception:
        return 1.0
    dc = cfg.get("decision_controller", {})
    try:
        return float(dc.get("dwell_threshold", 1.0))
    except Exception:
        return 1.0


DWELL_THRESHOLD = _load_dwell_threshold()


def _load_linear_constraints() -> tuple[list[list[float]], list[float]]:
    """Load linear constraint matrices ``A`` and vector ``b`` from config."""
    try:
        base = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(base, "config.yaml"), "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        dc = data.get("decision_controller", {}) if isinstance(data, dict) else {}
        lc = dc.get("linear_constraints", {}) if isinstance(dc, dict) else {}
        A = lc.get("A", []) or []
        b = lc.get("b", []) or []
        return A, b
    except Exception:
        return [], []


LINEAR_CONSTRAINTS_A, LINEAR_CONSTRAINTS_B = _load_linear_constraints()


def _load_watch_metrics() -> List[str]:
    """Load reporter metric paths to monitor from ``config.yaml``."""
    try:
        import yaml  # type: ignore

        base = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(base, "config.yaml"), "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        dc = data.get("decision_controller", {}) if isinstance(data, dict) else {}
        items = dc.get("watch_metrics", [])
        return [str(x) for x in items] if isinstance(items, list) else []
    except Exception:
        return []


def _load_watch_variables() -> List[str]:
    """Load module attribute paths to monitor from ``config.yaml``."""
    try:
        import yaml  # type: ignore

        base = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(base, "config.yaml"), "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        dc = data.get("decision_controller", {}) if isinstance(data, dict) else {}
        items = dc.get("watch_variables", [])
        return [str(x) for x in items] if isinstance(items, list) else []
    except Exception:
        return []


WATCH_METRICS = _load_watch_metrics()
WATCH_VARIABLES = _load_watch_variables()


def _load_phase_count() -> int:
    base = os.path.dirname(os.path.dirname(__file__))
    try:
        with open(os.path.join(base, "config.yaml"), "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception:
        return 1
    dc = data.get("decision_controller", {})
    try:
        return int(dc.get("phase_count", 1))
    except Exception:
        return 1


PHASE_COUNT = _load_phase_count()


def advance_step(cadence: int = CADENCE) -> bool:
    """Increment global step counter and enforce decision cadence."""
    global STEP_COUNTER
    STEP_COUNTER += 1
    cad = max(1, int(cadence))
    return STEP_COUNTER % cad == 0

# Track last state-change timestamp for each plugin
LAST_STATE_CHANGE: Dict[str, float] = {}


def record_plugin_state_change(name: str, now: float | None = None) -> None:
    """Record that ``name`` changed state at time ``now``."""
    if now is None:
        now = time.time()
    LAST_STATE_CHANGE[name] = float(now)


def tau_since_last_change(name: str, now: float | None = None) -> float:
    """Return seconds since ``name`` last changed state."""
    if now is None:
        now = time.time()
    last = LAST_STATE_CHANGE.get(name)
    if last is None:
        return float("inf")
    return float(now - last)


def update_dwell_counters(selected: Iterable[str], all_plugins: Iterable[str]) -> None:
    """Update consecutive-activation counts for ``all_plugins``."""
    sel = set(selected)
    for name in all_plugins:
        if name in sel:
            DWELL_COUNT[name] = DWELL_COUNT.get(name, 0) + 1
        else:
            DWELL_COUNT[name] = 0


def train_contribution_regressor(
    activation: torch.Tensor,
    outcomes: torch.Tensor,
    l1_penalty: float | None = None,
    lr: float = 0.01,
    epochs: int = 1000,
) -> torch.Tensor:
    """Train linear regressor ``q_a(z)`` with ℓ1 penalty.

    Parameters
    ----------
    activation:
        Matrix of plugin activations with shape ``(samples, plugins)``.
    outcomes:
        Target outcomes corresponding to each row in ``activation``.
    l1_penalty:
        Strength of the ℓ1 penalty. When ``None``, uses value from config.
    lr:
        Learning rate for simple gradient descent.
    epochs:
        Number of optimization steps.

    Returns
    -------
    torch.Tensor
        Learned weight vector of shape ``(plugins,)``.
    """

    if l1_penalty is None:
        l1_penalty = L1_PENALTY
    weights = torch.zeros(activation.shape[1], device=activation.device, requires_grad=True)
    for _ in range(epochs):
        pred = activation @ weights
        loss = ((pred - outcomes) ** 2).mean() + l1_penalty * weights.abs().sum()
        loss.backward()
        with torch.no_grad():
            weights -= lr * weights.grad
            weights.grad.zero_()
    return weights.detach()


class RewardSurrogate(nn.Module):
    """Simple quadratic surrogate ``q_omega(h, a)`` predicting rewards."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, 1, bias=False)
        self.pair = nn.Parameter(torch.zeros(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        lin = self.linear(x)
        pair = (x @ self.pair * x).sum(dim=1, keepdim=True)
        return (lin + pair).squeeze(1)


def train_reward_surrogate(
    activation: torch.Tensor,
    outcomes: torch.Tensor,
    lr: float = 0.01,
    epochs: int = 1000,
) -> RewardSurrogate:
    """Fit :class:`RewardSurrogate` to ``(activation, outcomes)`` pairs."""

    model = RewardSurrogate(activation.shape[1]).to(activation.device)
    for _ in range(epochs):
        pred = model(activation)
        loss = ((pred - outcomes) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= lr * p.grad
                p.grad.zero_()
            model.pair.fill_diagonal_(0.0)
    return model


def estimate_plugin_contributions(
    activation: torch.Tensor,
    outcomes: torch.Tensor,
    plugin_names: List[str],
    l1_penalty: float | None = None,
) -> Dict[str, float]:
    """Estimate per-plugin contribution scores.

    The contribution score for each plugin corresponds to the learned weight
    of the ℓ1-regularized regressor.
    """

    weights = train_contribution_regressor(activation, outcomes, l1_penalty)
    return {
        name: float(w.detach().to("cpu").item()) for name, w in zip(plugin_names, weights)
    }


def estimate_plugin_contributions_bayesian(
    activation: torch.Tensor,
    outcomes: torch.Tensor,
    plugin_names: List[str],
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Dict[str, tuple[float, float]]:
    """Estimate contribution mean and variance via Bayesian linear regression.

    A normal prior with precision ``alpha`` and noise precision ``beta`` is
    assumed.  The posterior over weights is returned as a mapping from plugin
    name to ``(mean, variance)``.
    """

    n, d = activation.shape
    eye = torch.eye(d, device=activation.device) * alpha
    precision = eye + beta * activation.t() @ activation
    cov = torch.inverse(precision)
    mean = beta * cov @ activation.t() @ outcomes
    out: Dict[str, tuple[float, float]] = {}
    for i, name in enumerate(plugin_names):
        m = float(mean[i].detach().to("cpu").item())
        v = float(cov[i, i].detach().to("cpu").item())
        out[name] = (m, v)
    return out


def get_plugin_cost(name: str) -> float:
    """Return intrinsic cost for ``name`` by inspecting its plugin module."""

    # Prefer dynamic measurements from ``plugin_cost_profiler``.
    prof_cost = _pcp.get_cost(name)
    if not math.isnan(prof_cost):
        return prof_cost

    try:
        mod = importlib.import_module(f"marble.plugins.{name}")
    except Exception:
        _pcp.record(name, 0.0)
        return 0.0

    for attr in ("PLUGIN_COST", "COST"):
        if hasattr(mod, attr):
            try:
                cost = float(getattr(mod, attr))
            except Exception:
                cost = 0.0
            _pcp.record(name, 0.0)
            return cost
    try:
        fn = getattr(mod, "estimate_cost")
        cost = float(fn())
    except Exception:
        cost = 0.0
    _pcp.record(name, 0.0)
    return cost


def decide_actions(
    h_t: Dict[str, Dict[str, float]],
    x_t: Dict[str, Any],
    history: Iterable[Dict[str, Any]],
    contrib_scores: Dict[str, float] | None = None,
    *,
    all_plugins: Iterable[str] | None = None,
    cost_recorder: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Select actions while enforcing incompatibilities, capacity and budget.

    Parameters
    ----------
    h_t:
        Mapping of plugin names to metadata containing at least a ``cost`` key.
    x_t:
        Proposed actions for plugins, typically produced by a planner.
    history:
        Iterable of previous action dictionaries. Used to enforce capacity
        limits across time.
    contrib_scores:
        Optional mapping of plugin names to contribution scores. Higher scores
        effectively reduce the plugin's cost during selection.

    Returns
    -------
    Dict[str, Any]
        Selected subset of ``x_t`` satisfying all constraints.
    """

    usage: Dict[str, int] = {}
    running_costs: Dict[str, float] = {}
    for past in history:
        for name, val in past.items():
            usage[name] = usage.get(name, 0) + 1
            if isinstance(val, dict) and "cost" in val:
                c = float(val.get("cost", 0.0))
            else:
                base = h_t.get(name, {})
                c = float(base.get("cost", get_plugin_cost(name)))
            running_costs[name] = running_costs.get(name, 0.0) + c

    now = time.time()

    # Ensure all candidate actions carry an explicit cost value. When the
    # caller omits ``h_t[name]['cost']`` we fall back to the plugin's intrinsic
    # cost and store it in ``PLUGIN_COST_CACHE`` for future decisions.
    for name in x_t:
        info = h_t.setdefault(name, {})
        if "cost" not in info:
            c = PLUGIN_COST_CACHE.get(name)
            if c is None:
                c = get_plugin_cost(name)
            info["cost"] = c
            PLUGIN_COST_CACHE[name] = c

    plugin_list = sorted(all_plugins or x_t.keys())
    name_to_idx = {n: i for i, n in enumerate(plugin_list)}
    action_vec = [0.0] * len(plugin_list)

    def penalty(name: str) -> float:
        tau = tau_since_last_change(name, now)
        if tau < TAU_THRESHOLD:
            return TAU_THRESHOLD - tau
        return 0.0

    # Sort candidates by effective cost including penalty so recent state changes
    # are deprioritized under the budget constraint
    ordered = sorted(
        x_t.items(),
        key=lambda kv: h_t.get(kv[0], {}).get("cost", 0.0)
        + penalty(kv[0])
        - (contrib_scores.get(kv[0], 0.0) if contrib_scores else 0.0)
        - DWELL_BONUS * DWELL_COUNT.get(kv[0], 0),
    )

    selected: Dict[str, Any] = {}
    active: Set[str] = set()
    remaining = BUDGET_LIMIT

    for name, action in ordered:
        base_cost = float(h_t.get(name, {}).get("cost", get_plugin_cost(name)))
        if contrib_scores:
            disc_cost = base_cost - float(contrib_scores.get(name, 0.0))
            disc_cost = max(disc_cost, 0.0)
        else:
            disc_cost = base_cost
        pen = penalty(name)
        real_cost = disc_cost + pen
        if not check_throughput(name, usage, CAPACITY_LIMITS):
            continue
        if not check_budget(name, real_cost, remaining, running_costs, BUDGET_LIMIT):
            continue
        if not check_incompatibility(name, active, INCOMPATIBILITY_SETS):
            continue
        idx = name_to_idx.get(name)
        if idx is not None:
            tentative = action_vec.copy()
            tentative[idx] = 1.0
            if not check_linear_constraints(
                tentative, LINEAR_CONSTRAINTS_A, LINEAR_CONSTRAINTS_B
            ):
                continue
        selected[name] = action
        active.add(name)
        usage[name] = usage.get(name, 0) + 1
        running_costs[name] = running_costs.get(name, 0.0) + real_cost
        if cost_recorder is not None:
            cost_recorder[name] = real_cost
        # Record the actual cost in the cache so subsequent calls have
        # up-to-date values even if the caller omits them.
        PLUGIN_COST_CACHE[name] = real_cost
        remaining = max(0.0, remaining - real_cost)
        if idx is not None:
            action_vec[idx] = 1.0
        if remaining <= 0:
            break
    return selected


class DecisionController:
    """High-level orchestrator integrating embeddings, sampling and learning."""

    def __init__(
        self,
        *,
        encoder: PluginEncoder | None = None,
        cadence: int = CADENCE,
        sampler_mode: str = "gumbel-top-k",
        top_k: int = 1,
        dwell_threshold: float = DWELL_THRESHOLD,
        lambda_lr: float = LAMBDA_LR,
        phase_count: int = PHASE_COUNT,
        watch_metrics: Iterable[str] | None = None,
        watch_variables: Iterable[str] | None = None,
        policy_mode: str = POLICY_MODE,
    ) -> None:
        if encoder is None:
            encoder = PluginEncoder(len(PLUGIN_ID_REGISTRY))
        self.encoder = encoder
        self.cadence = max(1, int(cadence))
        self.sampler_mode = sampler_mode
        self.top_k = int(top_k)
        self._last_phase: str | None = None
        self.lambda_lr = float(lambda_lr)
        self.phase_count = max(1, int(phase_count))
        self.policy_mode = policy_mode
        self.dwell_threshold = float(dwell_threshold)
        feat_dim = (
            self.encoder.embedding.embedding_dim
            + self.encoder.ctx_rnn.hidden_size
            + self.encoder.action_embed.embedding_dim
        )

        self.history_encoder = HistoryEncoder(
            feat_dim, len(PLUGIN_ID_REGISTRY), feat_dim
        )
        self._h_t: torch.Tensor | None = None
        self._prev_action_vec = torch.zeros(len(PLUGIN_ID_REGISTRY), dtype=torch.float32)
        self._prev_reward = 0.0
        self._steps_since_change = int(self.dwell_threshold)

        # ``cost_vec`` stores the per-plugin cost used by the policy-gradient
        # agent's budget constraint.  It is populated in :meth:`decide` from
        # the actual ``h_t[name]['cost']`` values supplied for the current
        # decision step so that penalties reflect real, up-to-date spending.
        self.cost_vec = torch.zeros(
            len(PLUGIN_ID_REGISTRY), dtype=torch.float32
        )

        def g_budget(actions: torch.Tensor) -> torch.Tensor:
            """Return normalised cost for the selected ``actions``."""

            return self.cost_vec[actions] / max(1.0, BUDGET_LIMIT)

        def g_dwell(actions: torch.Tensor) -> torch.Tensor:
            """Return dwell penalty based on action changes."""
            curr = torch.zeros_like(self._prev_action_vec)
            curr[actions] = 1.0
            diff = torch.abs(curr - self._prev_action_vec).sum()
            if self._steps_since_change < self.dwell_threshold:
                return diff.repeat(actions.shape[0])
            return torch.zeros_like(actions, dtype=torch.float32)

        self.agent = PolicyGradientAgent(
            state_dim=feat_dim,
            action_dim=len(PLUGIN_ID_REGISTRY),
            lambdas=[1.0, 1.0],
            constraints=[g_budget, g_dwell],
            lambda_lr=self.lambda_lr,
        )
        self.bayesian: BayesianPolicy | None = None
        if self.policy_mode == "bayesian":
            self.bayesian = BayesianPolicy(feat_dim, len(PLUGIN_ID_REGISTRY))
        self.reward_shaper = RewardShaper()
        self._reward_base = torch.tensor(
            [
                self.reward_shaper.w1,
                self.reward_shaper.w2,
                self.reward_shaper.w3,
                self.reward_shaper.w4,
                self.reward_shaper.w5,
                self.reward_shaper.w6,
            ],
            dtype=torch.float32,
        )
        if self.phase_count > 1:
            self.phase_proj = nn.Linear(feat_dim, self.phase_count)
            self.phase_bias = nn.Linear(
                self.phase_count, len(PLUGIN_ID_REGISTRY), bias=False
            )
            self.reward_phase = nn.Linear(self.phase_count, 6, bias=False)
        else:
            self.phase_proj = None
            self.phase_bias = None
            self.reward_phase = None
        self.trajectory = Trajectory()
        self.history: List[Dict[str, Any]] = []
        self.past_actions: List[str] = []
        self._metric_window = deque(maxlen=self.reward_shaper.window_size)
        self._prev_action_mask: Dict[str, int] = {}
        self.watch_metrics = set(watch_metrics if watch_metrics is not None else WATCH_METRICS)
        self.watch_variables = set(
            watch_variables if watch_variables is not None else WATCH_VARIABLES
        )
        # Store the most recent metrics observed via ``decide`` so callers can
        # inspect them even before the first decision step has been executed.
        # ``decide`` updates this mapping in-place when new metrics arrive.
        self.last_metrics: Dict[str, float] = {}
        self.divergence: bool = False
        # Logs used by the contribution regressor. Each entry in
        # ``_activation_log`` is a binary vector of length equal to the number
        # of registered plugins indicating which plugins were active during a
        # decision step. ``_reward_log`` stores the shaped reward observed after
        # that step.  These histories allow the controller to estimate per-
        # plugin contribution scores and feed them back into future selections.
        self._activation_log: List[torch.Tensor] = []
        self._reward_log: List[float] = []

    # --------------------------------------------------------------
    def _advance(self) -> bool:
        return advance_step(self.cadence)

    # --------------------------------------------------------------
    def _gather_watchables(self) -> Dict[str, float]:
        """Collect watched metrics and variables."""

        observed: Dict[str, float] = {}
        for path in self.watch_variables:
            try:
                mod_name, attr = path.rsplit(".", 1)
                mod = importlib.import_module(mod_name)
                val = getattr(mod, attr, None)
                if isinstance(val, (int, float)):
                    observed[path] = float(val)
            except Exception:
                continue
        for path in self.watch_metrics:
            parts = path.split("/")
            if len(parts) < 2:
                continue
            item = parts[-1]
            groups = parts[:-1]
            try:
                val = REPORTER.item(item, groups[0], *groups[1:])
                if isinstance(val, (int, float)):
                    observed[path] = float(val)
            except Exception:
                continue
        return observed

    # --------------------------------------------------------------
    def decide(
        self,
        h_t: Dict[str, Dict[str, float]],
        ctx_seq: torch.Tensor,
        *,
        metrics: Dict[str, float] | None = None,
    ) -> Dict[str, Any]:
        """Return selected actions using embeddings and stochastic policy.

        Per-call costs from ``h_t[name]["cost"]`` populate the controller's
        internal ``cost_vec`` so the soft budget constraint always reflects the
        current step's spending.
        """

        if not self._advance():
            self.history.append({})
            return {}

        watch_vals = self._gather_watchables()
        # Merge caller-provided metrics with values gathered from the
        # ``watch_*`` configuration.  When no metrics are available we keep
        # ``metrics`` as ``None`` so that reward shaping and history updates can
        # be skipped entirely.
        if not watch_vals and not metrics:
            metrics = None
        else:
            metrics = {**watch_vals, **(metrics or {})}
        if metrics is not None:
            self.last_metrics = metrics

        def _has_invalid(d: Dict[str, Any] | None) -> bool:
            if not d:
                return False
            for v in d.values():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    return True
            return False

        self.divergence = False
        if _has_invalid(metrics):
            self.divergence = True
        else:
            for info in h_t.values():
                if _has_invalid(info):
                    self.divergence = True
                    break
        if self.divergence:
            self._metric_window = deque(maxlen=self.reward_shaper.window_size)
            try:
                REPORTER.item[("reward_penalty", "decision_controller")] = math.inf
            except Exception:
                pass

        plugin_names = list(h_t.keys())
        # Refresh ``cost_vec`` using the per-step costs provided in ``h_t`` so
        # budget penalties and sampling reflect the latest spending.
        new_cost_vec = torch.zeros(len(PLUGIN_ID_REGISTRY), dtype=torch.float32)
        for name, info in h_t.items():
            idx = PLUGIN_ID_REGISTRY.get(name)
            if idx is not None:
                new_cost_vec[idx] = float(info.get("cost", 0.0))
        self.cost_vec = new_cost_vec
        ready = set(PLUGIN_GRAPH.recommend_next_plugin(self._last_phase))
        # If the plugin graph still contains pending nodes but none are ready
        # to run, we return an empty selection instead of considering all
        # plugins.  This prevents executing plugins whose prerequisites have
        # not yet been satisfied.
        if not ready and getattr(PLUGIN_GRAPH, "_deps", {}):
            self.history.append({})
            return {}
        if ready:
            plugin_names = [n for n in plugin_names if n in ready]
        if not plugin_names:
            self.history.append({})
            return {}
        plugin_ids = torch.tensor(
            [PLUGIN_ID_REGISTRY.get(n, 0) for n in plugin_names], dtype=torch.long
        )
        past_ids = [PLUGIN_ID_REGISTRY.get(n, 0) for n in self.past_actions] or [0]
        ctx_rep = ctx_seq.expand(len(plugin_ids), -1, -1)
        feats = self.encoder(plugin_ids, ctx_rep, past_ids)
        e_t = feats
        o_t = feats.mean(dim=0)
        r_prev = torch.tensor([self._prev_reward], dtype=o_t.dtype, device=o_t.device)
        self._h_t = self.history_encoder(o_t, self._prev_action_vec.to(o_t), r_prev, self._h_t)
        e_a_t = self._h_t[0].squeeze(0)
        if self.phase_proj is not None and self.phase_bias is not None:
            u = torch.softmax(self.phase_proj(e_a_t), dim=-1)
            phase_logits = self.phase_bias(u)
            mod = self.reward_phase(u) if self.reward_phase is not None else torch.zeros(6, dtype=e_a_t.dtype, device=e_a_t.device)
        else:
            u = torch.zeros(1, dtype=e_a_t.dtype, device=e_a_t.device)
            phase_logits = torch.zeros(len(PLUGIN_ID_REGISTRY), dtype=e_a_t.dtype, device=e_a_t.device)
            mod = torch.zeros(6, dtype=e_a_t.dtype, device=e_a_t.device)
        logits = compute_logits(e_t, e_a_t) + phase_logits[plugin_ids]
        for attr, base, m in zip(
            ["w1", "w2", "w3", "w4", "w5", "w6"],
            self._reward_base,
            mod,
        ):
            setattr(self.reward_shaper, attr, float((base + m).detach()))
        costs = self.cost_vec[plugin_ids]
        incompat: Dict[int, Set[int]] = {}
        name_to_idx = {n: i for i, n in enumerate(plugin_names)}
        for i, name in enumerate(plugin_names):
            others = {
                name_to_idx[o]
                for o in INCOMPATIBILITY_SETS.get(name, set())
                if o in name_to_idx
            }
            if others:
                incompat[i] = others
        if self.policy_mode == "bayesian" and self.bayesian is not None:
            theta = self.bayesian.sample(plugin_ids)
            scores = (e_t * theta).sum(dim=1)
            lam = self.agent.lambdas[0] if self.agent.lambdas else 0.0
            scores = scores - lam * costs
            idx = int(torch.argmax(scores))
            chosen = {plugin_ids[idx].item()}
            x_t = {
                plugin_names[idx]: h_t[plugin_names[idx]].get("action", "on")
            }
        else:
            mask = sample_actions(
                logits,
                mode=self.sampler_mode,
                top_k=self.top_k,
                temperature=1.0,
                costs=costs,
                budget=BUDGET_LIMIT,
                incompatibility=incompat,
            )
            indices = (mask > 0.5).nonzero(as_tuple=False).squeeze(1)
            chosen = plugin_ids[indices].tolist()
            x_t = {
                name: h_t[name].get("action", "on")
                for name in plugin_names
                if PLUGIN_ID_REGISTRY.get(name, -1) in chosen
            }

        contrib_scores = None
        if self._activation_log and self._reward_log:
            try:
                activation = torch.stack(self._activation_log)
                outcomes = torch.tensor(self._reward_log, dtype=torch.float32)
                # ``compute_contributions`` encapsulates both the classical
                # and Bayesian estimators defined in this module.  Using it
                # here avoids duplicating estimator selection logic in the
                # controller and makes future estimator extensions instantly
                # available to the decision loop.
                contrib_map = self.compute_contributions(
                    activation, outcomes, list(PLUGIN_ID_REGISTRY.keys())
                )
                contrib_scores = {
                    n: contrib_map["individual"].get(n, 0.0)
                    for n in plugin_names
                    if n in contrib_map["individual"]
                }
            except Exception:
                contrib_scores = None

        cost_recorder: Dict[str, float] = {}
        selected = decide_actions(
            h_t,
            x_t,
            self.history,
            contrib_scores=contrib_scores,
            all_plugins=plugin_names,
            cost_recorder=cost_recorder,
        )
        if selected:
            self.history.append(
                {
                    n: {"action": a, "cost": cost_recorder.get(n, 0.0)}
                    for n, a in selected.items()
                }
            )
        else:
            self.history.append({})
        self.past_actions.extend(selected.keys())
        if selected:
            self._last_phase = next(iter(selected))
        else:
            self._last_phase = None

        action_mask = {
            n: 1 if selected.get(n, "off") != "off" else 0 for n in plugin_names
        }
        delta_mask = {
            n: abs(action_mask.get(n, 0) - self._prev_action_mask.get(n, 0))
            for n in action_mask
        }
        delta_total = sum(delta_mask.values())
        if delta_total > 0 and self._steps_since_change < self.dwell_threshold:
            selected = {}
            action_mask = {n: 0 for n in plugin_names}
            delta_mask = {n: 0 for n in plugin_names}
            delta_total = 0
            self._steps_since_change += 1
        else:
            self._steps_since_change = (
                self._steps_since_change + 1 if delta_total == 0 else 0
            )
        self._prev_action_mask = action_mask

        reward = 0.0
        log_reward = False
        if metrics is not None and not self.divergence:
            cleaned: Dict[str, float] = {}
            for k, v in metrics.items():
                try:
                    f = float(v)
                except Exception:
                    continue
                if math.isnan(f) or math.isinf(f):
                    continue
                cleaned[k] = f
            if cleaned:
                self._metric_window.append(cleaned)
        if metrics is not None or self.divergence:
            window = list(self._metric_window)
            reward, _ = self.reward_shaper.update(
                window,
                action_mask,
                delta_mask,
                h_t,
                INCOMPATIBILITY_SETS,
                force_divergence=self.divergence,
            )
            log_reward = True

        now = time.time()
        if selected:
            for name in selected:
                record_plugin_state_change(name, now)
                PLUGIN_GRAPH.mark_executed(name)
        update_dwell_counters(selected.keys(), plugin_names)

        probs = torch.sigmoid(logits)
        if selected and log_reward:
            for name in selected:
                pid = PLUGIN_ID_REGISTRY.get(name, 0)
                if self.policy_mode == "bayesian":
                    prob = 1.0
                else:
                    prob = float(
                        probs[(plugin_ids == pid).nonzero(as_tuple=False)][0].detach()
                    )
                self.trajectory.log(pid, reward, prob, prob)

        if selected:
            acts = torch.tensor(
                [PLUGIN_ID_REGISTRY[n] for n in selected.keys()],
                dtype=torch.long,
            )
            if log_reward:
                state = e_a_t.unsqueeze(0).repeat(len(acts), 1)
                rets = torch.full((len(acts),), reward, dtype=state.dtype)
                if self.policy_mode == "policy-gradient":
                    self.agent.step(state, acts, rets)
                else:
                    for pid in acts.tolist():
                        phi = e_t[(plugin_ids == pid).nonzero(as_tuple=False)[0]]
                        if self.bayesian is not None:
                            self.bayesian.update(pid, phi, float(reward))
            self.agent.update_lambdas(acts)

        act_vec = torch.zeros(len(PLUGIN_ID_REGISTRY), dtype=torch.float32)
        for name in plugin_names:
            idx = PLUGIN_ID_REGISTRY.get(name)
            if idx is not None:
                act_vec[idx] = 1.0 if name in selected else 0.0
        if log_reward:
            self._activation_log.append(act_vec)
            self._reward_log.append(float(reward))
            self._prev_reward = float(reward)
            # Only advance the cached action vector when we also log a reward so
            # that ``_prev_action_vec`` and ``_prev_reward`` always describe the
            # same decision. Skipping this update keeps the pair aligned when
            # metrics are missing and no reward is produced.
            self._prev_action_vec = act_vec
        if isinstance(self._h_t, tuple):
            self._h_t = tuple(t.detach() for t in self._h_t)
        else:
            self._h_t = self._h_t.detach()

        return selected

    # --------------------------------------------------------------
    def offpolicy_value(self, q_hat: List[float]) -> float:
        """Return doubly-robust off-policy estimate for logged trajectory."""

        return doubly_robust(self.trajectory, q_hat)

    # --------------------------------------------------------------
    def compute_contributions(
        self,
        activation: torch.Tensor,
       outcomes: torch.Tensor,
       plugin_names: List[str],
        *,
        bayesian: bool = False,
    ) -> Dict[str, Any]:
        """Return individual and pairwise contribution estimates."""

        if bayesian:
            mean_var = estimate_plugin_contributions_bayesian(
                activation, outcomes, plugin_names
            )
            indiv = {k: v[0] for k, v in mean_var.items()}
            return {"individual": indiv, "pairwise": {}}

        surrogate = train_reward_surrogate(activation, outcomes)
        d = activation.shape[1]
        base = torch.zeros(d, device=activation.device)
        base_val = float(surrogate(base.unsqueeze(0)).detach().to("cpu").item())

        indiv: Dict[str, float] = {}
        preds = {}
        for i, name in enumerate(plugin_names):
            vec = base.clone(); vec[i] = 1.0
            val = float(surrogate(vec.unsqueeze(0)).detach().to("cpu").item())
            indiv[name] = val - base_val
            preds[i] = val

        pair: Dict[tuple[str, str], float] = {}
        for i in range(len(plugin_names)):
            for j in range(i + 1, len(plugin_names)):
                vec = base.clone(); vec[i] = 1.0; vec[j] = 1.0
                val = float(surrogate(vec.unsqueeze(0)).detach().to("cpu").item())
                delta = val - preds[i] - preds[j] + base_val
                pair[(plugin_names[i], plugin_names[j])] = delta

        return {"individual": indiv, "pairwise": pair}


__all__ = [
    "decide_actions",
    "INCOMPATIBILITY_SETS",
    "CAPACITY_LIMITS",
    "BUDGET_LIMIT",
    "train_contribution_regressor",
    "estimate_plugin_contributions",
    "estimate_plugin_contributions_bayesian",
    "L1_PENALTY",
    "TAU_THRESHOLD",
    "LAMBDA_LR",
    "CADENCE",
    "PHASE_COUNT",
    "POLICY_MODE",
    "STEP_COUNTER",
    "advance_step",
    "LAST_STATE_CHANGE",
    "record_plugin_state_change",
    "tau_since_last_change",
    "DWELL_BONUS",
    "DWELL_COUNT",
    "update_dwell_counters",
    "get_plugin_cost",
    "DecisionController",
]
