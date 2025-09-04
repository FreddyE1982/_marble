"""Policy gradient training helpers.

This module implements a lightweight policy gradient agent with
entropy regularization, a critic baseline, and action constraints.
It avoids ``torch.optim`` and ``torch.nn`` layers by operating directly
on tensors.
"""

from __future__ import annotations

from typing import Callable, Sequence, List

import torch

ConstraintFn = Callable[[torch.Tensor], torch.Tensor]


class PolicyGradientAgent:
    """Simple policy gradient agent with entropy bonus and constraints.

    Parameters
    ----------
    state_dim:
        Dimension of the input state vector.
    action_dim:
        Number of discrete actions.
    lr:
        Learning rate for the manual gradient step.
    beta:
        Weight for the entropy bonus.
    lambdas:
        Coefficients for constraint penalties.  ``lambdas[i]`` is paired with
        ``constraints[i]``.
    constraints:
        Sequence of callables ``g_j`` taking an action tensor and returning a
        penalty tensor of the same shape.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-2,
        beta: float = 0.01,
        lambdas: Sequence[float] | None = None,
        constraints: Sequence[ConstraintFn] | None = None,
        lambda_lr: float = 0.1,
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.lr = float(lr)
        self.beta = float(beta)
        self.lambdas: List[float] = list(lambdas) if lambdas is not None else []
        self.constraints: List[ConstraintFn] = list(constraints) if constraints is not None else []
        self.lambda_lr = float(lambda_lr)
        # Running averages of constraint violations ``g_j(a)``
        self._g_avg: List[float] = [0.0 for _ in self.constraints]
        self._g_count: List[int] = [0 for _ in self.constraints]

        # Policy parameters: linear logits ``s @ Wp + bp``
        self.Wp = torch.zeros(self.state_dim, self.action_dim, requires_grad=True)
        self.bp = torch.zeros(self.action_dim, requires_grad=True)
        # Critic parameters: value estimate ``s @ Wv + bv``
        self.Wv = torch.zeros(self.state_dim, 1, requires_grad=True)
        self.bv = torch.zeros(1, requires_grad=True)

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def _policy_logits(self, states: torch.Tensor) -> torch.Tensor:
        return states @ self.Wp + self.bp

    def _values(self, states: torch.Tensor) -> torch.Tensor:
        return (states @ self.Wv + self.bv).squeeze(-1)

    def action_probs(self, states: torch.Tensor) -> torch.Tensor:
        """Return action probabilities for each state."""
        return torch.softmax(self._policy_logits(states), dim=-1)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
    ) -> float:
        """Perform a single policy/critic update and return the loss."""
        probs = self.action_probs(states)
        dist = torch.distributions.Categorical(probs=probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        values = self._values(states)
        advantages = returns - values.detach()

        penalty = torch.zeros_like(log_probs)
        for lam, g in zip(self.lambdas, self.constraints):
            penalty = penalty + lam * g(actions)

        loss_pg = -(log_probs * (advantages - penalty) + self.beta * entropy)
        loss_critic = 0.5 * (returns - values) ** 2
        loss = (loss_pg + loss_critic).mean()

        for p in [self.Wp, self.bp, self.Wv, self.bv]:
            if p.grad is not None:
                p.grad.zero_()
        loss.backward()
        with torch.no_grad():
            for p in [self.Wp, self.bp, self.Wv, self.bv]:
                p -= self.lr * p.grad
        return loss.detach().to("cpu").item()

    # ------------------------------------------------------------------
    # Lagrange multiplier update
    # ------------------------------------------------------------------
    def lambda_updates(self, actions: torch.Tensor) -> None:
        """Update Lagrange multipliers based on constraint violations.

        Each constraint ``g_j`` produces a penalty term given the selected
        ``actions``.  We maintain a running mean of these penalties and adjust
        ``lambda_j`` via ``lambda_j += eta * mean(g_j(a))`` clipped to be
        non-negative.
        """

        if not self.constraints:
            return

        with torch.no_grad():
            for j, g in enumerate(self.constraints):
                g_val = g(actions).detach().to("cpu").mean().item()
                self._g_count[j] += 1
                count = self._g_count[j]
                avg = self._g_avg[j] + (g_val - self._g_avg[j]) / count
                self._g_avg[j] = avg
                new_lam = self.lambdas[j] + self.lambda_lr * avg
                self.lambdas[j] = max(0.0, new_lam)


__all__ = ["PolicyGradientAgent"]
