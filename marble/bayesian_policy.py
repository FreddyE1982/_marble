"""Bayesian linear policy for Thompson sampling.

This module maintains Gaussian posteriors over linear reward models for
individual actions.  For each action ``k`` we assume a weight vector
``theta_k`` with prior ``N(0, I)`` and observe rewards ``r`` generated via
``r = phi(h, a)^T theta_k + eps`` where ``eps ~ N(0, sigma^2)``.  The
posterior parameters ``mu_k`` and ``Sigma_k`` are updated in closed form
after every observation.  ``sample`` draws parameters for Thompson
sampling and ``update`` performs a recursive Bayesian update.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
from torch.distributions.multivariate_normal import MultivariateNormal

logger = logging.getLogger(__name__)


class BayesianPolicy:
    """Maintain per-action Gaussian posteriors for linear rewards."""

    def __init__(self, feat_dim: int, action_dim: int, noise_var: float = 1.0) -> None:
        self.feat_dim = int(feat_dim)
        self.action_dim = int(action_dim)
        self.noise_var = float(noise_var)
        eye = torch.eye(self.feat_dim, dtype=torch.float32)
        self.mu = torch.zeros(self.action_dim, self.feat_dim, dtype=torch.float32)
        self.cov = eye.expand(self.action_dim, -1, -1).clone()

    # --------------------------------------------------------------
    def sample(self, action_ids: torch.Tensor) -> torch.Tensor:
        """Return parameter samples for ``action_ids``.

        Parameters
        ----------
        action_ids:
            Tensor of action indices to sample for.
        """

        samples = []
        for idx in action_ids.tolist():
            dist = MultivariateNormal(self.mu[idx], self.cov[idx])
            samples.append(dist.sample())
        return torch.stack(samples)

    # --------------------------------------------------------------
    def update(self, action: int, phi: torch.Tensor, reward: float) -> None:
        """Update posterior for ``action`` using ``phi`` and ``reward``."""

        mu = self.mu[action]
        cov = self.cov[action]
        phi = phi.reshape(-1, 1)
        s_phi = cov @ phi
        denom = self.noise_var + (phi.t() @ s_phi).item()
        gain = s_phi / denom
        resid = reward - (phi.t() @ mu.reshape(-1, 1)).item()
        mu_new = mu + gain.flatten() * resid
        cov_new = cov - gain @ s_phi.t()
        self.mu[action] = mu_new
        self.cov[action] = cov_new
        logger.debug("action %s mu %s", action, mu_new)
        logger.debug("action %s Sigma diag %s", action, cov_new.diag())

    # --------------------------------------------------------------
    def get_posterior(self, action: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return current ``(mu, cov)`` for ``action``."""

        return self.mu[action].clone(), self.cov[action].clone()


__all__ = ["BayesianPolicy"]
