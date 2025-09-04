"""Reward shaping based on performance trends.

Maintains a sliding window of latency, throughput and cost statistics. For each
window, Ordinary Least Squares (OLS) is used to estimate the linear trend of
these metrics over time. The resulting slopes are combined into a shaped reward
suited for actor–critic updates: decreasing latency and cost yield positive
rewards while increasing throughput is rewarded.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Tuple
import os
import torch


def _config_window_size() -> int:
    """Load ``reward_shaper.window_size`` from ``config.yaml``.

    Returns a value of at least ``2``. Falls back to ``10`` if the file or entry
    is missing or invalid.
    """

    try:
        base = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(base, "config.yaml"), "r", encoding="utf-8") as fh:
            in_section = False
            for raw in fh:
                line = raw.split("#", 1)[0].strip().lower()
                if not line:
                    continue
                if line.startswith("reward_shaper:"):
                    in_section = True
                    continue
                if in_section:
                    if line.startswith("window_size:"):
                        try:
                            val = int(line.split(":", 1)[1])
                            return max(2, val)
                        except Exception:
                            return 10
                    if line[0] not in (" ", "\t"):
                        break
    except Exception:
        pass
    return 10


class RewardShaper:
    """Track performance metrics and compute shaped rewards.

    Parameters
    ----------
    window_size:
        Number of recent samples kept for trend estimation. Defaults to the
        ``reward_shaper.window_size`` value from :mod:`config.yaml`.
    """

    def __init__(self, window_size: int | None = None) -> None:
        size = window_size if window_size is not None else _config_window_size()
        self.window_size = max(2, int(size))
        self._lat: Deque[float] = deque(maxlen=self.window_size)
        self._thr: Deque[float] = deque(maxlen=self.window_size)
        self._cost: Deque[float] = deque(maxlen=self.window_size)

    @staticmethod
    def _ols_beta(values: Deque[float]) -> float:
        """Return the slope of ``values`` over their indices using OLS.

        The computation is performed using ``torch`` tensors to comply with the
        repository policy that torch is imported when numerical work occurs.
        """

        n = len(values)
        if n < 2:
            return 0.0
        x = torch.arange(float(n))
        y = torch.tensor(list(values), dtype=torch.float32)
        x_mean = x.mean()
        y_mean = y.mean()
        cov = ((x - x_mean) * (y - y_mean)).sum()
        var = ((x - x_mean) ** 2).sum()
        beta = cov / var if var != 0 else torch.tensor(0.0)
        return float(beta)

    def update(self, latency: float, throughput: float, cost: float) -> Tuple[float, Dict[str, float]]:
        """Add a new observation and return shaped reward and betas.

        Parameters
        ----------
        latency, throughput, cost:
            Latest performance statistics.

        Returns
        -------
        reward, betas:
            A tuple containing the shaped reward and a dictionary mapping metric
            names to their corresponding OLS slopes.
        """

        self._lat.append(float(latency))
        self._thr.append(float(throughput))
        self._cost.append(float(cost))

        beta_lat = self._ols_beta(self._lat)
        beta_thr = self._ols_beta(self._thr)
        beta_cost = self._ols_beta(self._cost)

        reward = -beta_lat + beta_thr - beta_cost
        betas = {"latency": beta_lat, "throughput": beta_thr, "cost": beta_cost}
        return reward, betas


__all__ = ["RewardShaper"]
