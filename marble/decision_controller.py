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
from typing import Any, Dict, Iterable, List, Set

import torch

from .constraints import check_budget, check_incompatibility, check_throughput
from .plugin_graph import PLUGIN_GRAPH
from .plugin_encoder import PluginEncoder
from .action_sampler import compute_logits, select_plugins
from .reward_shaper import RewardShaper
from .offpolicy import Trajectory, doubly_robust
from .policy_gradient import PolicyGradientAgent
from .plugins import PLUGIN_ID_REGISTRY
from .reporter import REPORTER

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


def _load_budget() -> float:
    """Load budget limit from ``config.yaml``."""
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
        return float("inf")
    dc = cfg.get("decision_controller", {})
    try:
        return float(dc.get("budget", float("inf")))
    except Exception:
        return float("inf")


BUDGET_LIMIT = _load_budget()


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
DWELL_COUNT: Dict[str, int] = {}


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

    try:
        mod = importlib.import_module(f"marble.plugins.{name}")
    except Exception:
        return 0.0

    for attr in ("PLUGIN_COST", "COST"):
        if hasattr(mod, attr):
            try:
                return float(getattr(mod, attr))
            except Exception:
                return 0.0
    try:
        fn = getattr(mod, "estimate_cost")
        return float(fn())
    except Exception:
        return 0.0


def decide_actions(
    h_t: Dict[str, Dict[str, float]],
    x_t: Dict[str, Any],
    history: Iterable[Dict[str, Any]],
    contrib_scores: Dict[str, float] | None = None,
    *,
    all_plugins: Iterable[str] | None = None,
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
        for name in past:
            usage[name] = usage.get(name, 0) + 1
            base = h_t.get(name, {})
            c = float(base.get("cost", get_plugin_cost(name)))
            running_costs[name] = running_costs.get(name, 0.0) + c

    now = time.time()

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
        pen = penalty(name)
        cost = base_cost + pen
        if contrib_scores:
            cost -= float(contrib_scores.get(name, 0.0))
        if not check_throughput(name, usage, CAPACITY_LIMITS):
            continue
        if not check_budget(name, cost, remaining, running_costs, BUDGET_LIMIT):
            continue
        if not check_incompatibility(name, active, INCOMPATIBILITY_SETS):
            continue
        selected[name] = action
        active.add(name)
        usage[name] = usage.get(name, 0) + 1
        running_costs[name] = running_costs.get(name, 0.0) + cost
        remaining -= cost
        record_plugin_state_change(name, now)
        PLUGIN_GRAPH.mark_executed(name)
        if remaining <= 0:
            break
    update_dwell_counters(selected.keys(), all_plugins or x_t.keys())
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
        watch_metrics: Iterable[str] | None = None,
        watch_variables: Iterable[str] | None = None,
    ) -> None:
        if encoder is None:
            encoder = PluginEncoder(len(PLUGIN_ID_REGISTRY))
        self.encoder = encoder
        self.cadence = max(1, int(cadence))
        self.sampler_mode = sampler_mode
        self.top_k = int(top_k)
        self._last_phase: str | None = None
        feat_dim = (
            self.encoder.embedding.embedding_dim
            + self.encoder.ctx_rnn.hidden_size
            + self.encoder.action_embed.embedding_dim
        )

        # Build a cost vector so the policy-gradient agent can impose a soft
        # budget constraint through its generic ``constraints`` interface.
        cost_vec = torch.tensor(
            [get_plugin_cost(n) for n in PLUGIN_ID_REGISTRY.keys()],
            dtype=torch.float32,
        )

        def g_budget(actions: torch.Tensor) -> torch.Tensor:
            """Return normalised cost for the selected ``actions``."""

            return cost_vec[actions] / max(1.0, BUDGET_LIMIT)

        self.agent = PolicyGradientAgent(
            state_dim=feat_dim,
            action_dim=len(PLUGIN_ID_REGISTRY),
            lambdas=[1.0],
            constraints=[g_budget],
        )
        self.reward_shaper = RewardShaper()
        self.trajectory = Trajectory()
        self.history: List[Dict[str, Any]] = []
        self.past_actions: List[str] = []
        self.watch_metrics = set(watch_metrics if watch_metrics is not None else WATCH_METRICS)
        self.watch_variables = set(
            watch_variables if watch_variables is not None else WATCH_VARIABLES
        )
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
        """Return selected actions using embeddings and stochastic policy."""

        if not self._advance():
            self.history.append({})
            return {}

        watch_vals = self._gather_watchables()
        # Merge caller-provided metrics with values gathered from the
        # ``watch_*`` configuration.  When no metrics are available we keep an
        # empty dict so the caller can still inspect ``last_metrics`` without
        # triggering reward shaping on placeholder zeros.
        metrics = {**watch_vals, **(metrics or {})}
        self.last_metrics = metrics

        plugin_names = list(h_t.keys())
        ready = set(PLUGIN_GRAPH.recommend_next_plugin(self._last_phase))
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
        e_a_t = feats.mean(dim=0)
        logits = compute_logits(e_t, e_a_t)
        chosen = select_plugins(
            plugin_ids, e_t, e_a_t, mode=self.sampler_mode, top_k=self.top_k
        )
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
                all_names = list(PLUGIN_ID_REGISTRY.keys())
                contrib_map = estimate_plugin_contributions(
                    activation, outcomes, all_names
                )
                contrib_scores = {
                    n: contrib_map.get(n, 0.0) for n in plugin_names if n in contrib_map
                }
            except Exception:
                contrib_scores = None

        selected = decide_actions(
            h_t,
            x_t,
            self.history,
            contrib_scores=contrib_scores,
            all_plugins=plugin_names,
        )
        self.history.append(selected)
        self.past_actions.extend(selected.keys())
        if selected:
            self._last_phase = next(iter(selected))
        else:
            self._last_phase = None

        reward = 0.0
        if metrics:
            reward, _ = self.reward_shaper.update(
                metrics.get("latency", 0.0),
                metrics.get("throughput", 0.0),
                metrics.get("cost", 0.0),
            )

        probs = torch.sigmoid(logits)
        for name in selected:
            pid = PLUGIN_ID_REGISTRY.get(name, 0)
            prob = float(probs[(plugin_ids == pid).nonzero(as_tuple=False)][0].detach())
            self.trajectory.log(pid, reward, prob, prob)

        if selected:
            state = e_a_t.unsqueeze(0)
            act = torch.tensor([PLUGIN_ID_REGISTRY[list(selected.keys())[0]]])
            ret = torch.tensor([reward])
            self.agent.step(state, act, ret)

        act_vec = torch.zeros(len(PLUGIN_ID_REGISTRY), dtype=torch.float32)
        for name in plugin_names:
            idx = PLUGIN_ID_REGISTRY.get(name)
            if idx is not None:
                act_vec[idx] = 1.0 if name in selected else 0.0
        self._activation_log.append(act_vec)
        self._reward_log.append(float(reward))

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
        """Wrapper around contribution estimators."""

        if bayesian:
            return estimate_plugin_contributions_bayesian(
                activation, outcomes, plugin_names
            )
        return estimate_plugin_contributions(activation, outcomes, plugin_names)


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
    "CADENCE",
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
