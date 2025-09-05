"""Reward shaping based on performance trends.

Maintains a sliding window of latency, throughput and cost statistics. For each
window, Ordinary Least Squares (OLS) is used to estimate the linear trend of
these metrics over time. The resulting slopes are combined into a shaped reward
suited for actor–critic updates: decreasing latency and cost yield positive
rewards while increasing throughput is rewarded.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import os
import torch


def _load_reward_config() -> Dict[str, float]:
    """Load reward-shaper parameters from ``config.yaml``.

    The configuration section ``reward_shaper`` may define ``window_size``,
    weights ``w1`` through ``w6`` and the divergence threshold ``M_div``. Missing
    entries fall back to sensible defaults. ``window_size`` is clamped to at
    least ``2`` to ensure meaningful slope estimates.
    """

    defaults: Dict[str, float] = {
        "window_size": 10,
        "w1": 1.0,
        "w2": 1.0,
        "w3": 1.0,
        "w4": 1.0,
        "w5": 1.0,
        "w6": 1.0,
        "M_div": 0.1,
    }
    try:
        base = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(base, "config.yaml"), "r", encoding="utf-8") as fh:
            section: str | None = None
            for raw in fh:
                line = raw.split("#", 1)[0].rstrip()
                if not line:
                    continue
                if not line.startswith(" ") and line.endswith(":"):
                    section = line[:-1].strip()
                    continue
                if section == "reward_shaper" and ":" in line:
                    k, v = line.split(":", 1)
                    try:
                        defaults[k.strip()] = float(v.strip())
                    except Exception:
                        continue
    except Exception:
        pass
    defaults["window_size"] = max(2, int(defaults.get("window_size", 10)))
    return defaults


class RewardShaper:
    """Track performance metrics and compute shaped rewards.

    Parameters
    ----------
    window_size:
        Number of recent samples used when computing slopes. Defaults to the
        ``reward_shaper.window_size`` value from :mod:`config.yaml`.
    w1..w6:
        Optional overrides for reward weights. When ``None`` the corresponding
        value from configuration is used.
    M_div:
        Divergence threshold for the throughput-normalised drop indicator.
    """

    def __init__(
        self,
        window_size: int | None = None,
        *,
        w1: float | None = None,
        w2: float | None = None,
        w3: float | None = None,
        w4: float | None = None,
        w5: float | None = None,
        w6: float | None = None,
        M_div: float | None = None,
    ) -> None:
        cfg = _load_reward_config()
        size = window_size if window_size is not None else cfg["window_size"]
        self.window_size = max(2, int(size))
        self.w1 = float(cfg["w1"] if w1 is None else w1)
        self.w2 = float(cfg["w2"] if w2 is None else w2)
        self.w3 = float(cfg["w3"] if w3 is None else w3)
        self.w4 = float(cfg["w4"] if w4 is None else w4)
        self.w5 = float(cfg["w5"] if w5 is None else w5)
        self.w6 = float(cfg["w6"] if w6 is None else w6)
        self.M_div = float(cfg["M_div"] if M_div is None else M_div)

    @staticmethod
    def _ols_beta(x_vals: List[float], y_vals: List[float]) -> float:
        """Return the slope of ``y_vals`` over ``x_vals`` using OLS."""

        n = len(x_vals)
        if n < 2:
            return 0.0
        x = torch.tensor(x_vals, dtype=torch.float32)
        y = torch.tensor(y_vals, dtype=torch.float32)
        x_mean = x.mean()
        y_mean = y.mean()
        cov = ((x - x_mean) * (y - y_mean)).sum()
        var = ((x - x_mean) ** 2).sum()
        beta = cov / var if var != 0 else torch.tensor(0.0)
        return float(beta)

    @staticmethod
    def _ema(values: List[float], alpha: float = 0.5) -> float:
        """Return the exponential moving average of ``values``."""

        if not values:
            return 0.0
        ema = float(values[0])
        for v in values[1:]:
            ema = alpha * float(v) + (1.0 - alpha) * ema
        return ema

    def update(
        self,
        window: List[Dict[str, float]],
        action_mask: Dict[str, int],
        action_deltas: Dict[str, int],
        h_t: Dict[str, Dict[str, float]],
        incompat: Dict[str, set] | None = None,
        *,
        force_divergence: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute shaped reward from ``window`` and action information.

        Parameters
        ----------
        window:
            List of recent metric dictionaries. Each entry may contain
            ``latency``, ``throughput``, ``cost`` and ``wall_time``.
        action_mask:
            Mapping of plugin names to ``1`` if active in the current step.
        action_deltas:
            Mapping of plugin names to ``1`` when their activation state toggled
            in this step.
        h_t:
            Plugin metadata containing at least per-plugin ``cost`` values.
        incompat:
            Optional incompatibility mapping used to penalise conflicting
            activations.
        force_divergence:
            When ``True`` the update bypasses slope calculations and returns a
            fixed penalty of ``-M_div`` regardless of the provided metrics.
        """

        incompat = incompat or {}
        if force_divergence:
            reward = -self.M_div
            components = {
                "latency_slope": 0.0,
                "throughput_slope": 0.0,
                "cost_slope": 0.0,
                "throughput_drop": self.M_div,
                "divergence": 1.0,
                "toggle_penalty": 0.0,
                "compatibility_penalty": 0.0,
                "compute_cost_penalty": 0.0,
                "reward": reward,
            }
            return reward, components
        lat = [m.get("latency", 0.0) for m in window]
        thr = [m.get("throughput", 0.0) for m in window]
        cst = [m.get("cost", 0.0) for m in window]
        times = [m.get("wall_time", i) for i, m in enumerate(window)]

        ema_thr = self._ema(thr)
        beta_lat = self._ols_beta(times, lat)
        beta_thr = self._ols_beta(times, thr)
        beta_cost = self._ols_beta(times, cst)

        drop = 0.0
        if ema_thr > 0.0:
            drop = max(0.0, -beta_thr) / max(ema_thr, 1e-6)
        divergence = 1.0 if drop > self.M_div else 0.0

        toggle_pen = float(sum(abs(v) for v in action_deltas.values()))
        active = [n for n, v in action_mask.items() if v]
        compat_pen = 0.0
        for i, a in enumerate(active):
            for b in active[i + 1 :]:
                if b in incompat.get(a, set()):
                    compat_pen += 1.0
        compute_pen = float(sum(h_t.get(n, {}).get("cost", 0.0) for n in active))
        total_pen = toggle_pen + compat_pen + compute_pen

        reward = (
            self.w1 * (-beta_lat)
            + self.w2 * beta_thr
            + self.w3 * (-beta_cost)
            + self.w4 * (-drop)
            + self.w5 * (-divergence)
            + self.w6 * (-total_pen)
        )

        components = {
            "latency_slope": beta_lat,
            "throughput_slope": beta_thr,
            "cost_slope": beta_cost,
            "throughput_drop": drop,
            "divergence": divergence,
            "toggle_penalty": toggle_pen,
            "compatibility_penalty": compat_pen,
            "compute_cost_penalty": compute_pen,
            "reward": reward,
        }
        return reward, components


__all__ = ["RewardShaper"]
