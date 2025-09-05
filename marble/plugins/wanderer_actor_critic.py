from __future__ import annotations

r"""Actor-Critic based Wanderer plugin.

This plugin maintains tiny actor and critic models built directly with
:mod:`torch` tensor operations.  At each step it computes a discounted
return ::

    G = sum_k gamma^k * [-w_L * l_t + w_beta * beta_t + w_d * d_{a_t}]

Gradients flow through the actor, critic and the learnable weighting
factors.  Parameter updates are applied manually without relying on
``torch.optim`` as required by project rules.  Actions are sampled from
probabilities over the provided action set so the selected action always
respects ``a_t \in A``.
"""

import torch
from typing import Any, List, Tuple

from ..wanderer import expose_learnable_params


class ActorCriticPlugin:
    """Minimal actor–critic Wanderer plugin."""

    @staticmethod
    @expose_learnable_params
    def _params(
        wanderer,
        *,
        w_loss: float = 1.0,
        w_beta: float = 1.0,
        w_dist: float = 1.0,
        gamma: float = 0.9,
    ):
        return w_loss, w_beta, w_dist, gamma

    def on_init(self, wanderer) -> None:
        device = getattr(wanderer, "_device", "cpu")
        self.actor_w = torch.zeros(1, requires_grad=True, device=device)
        self.critic_w = torch.zeros(1, requires_grad=True, device=device)
        self.lr = 0.01
        self._ret = torch.tensor(0.0, device=device)
        self._last_logprob = torch.tensor(0.0, device=device)

    def choose_next(
        self, wanderer, current, choices: List[Tuple["Synapse", str]]
    ):
        if not choices:
            return None, "forward"
        device = getattr(wanderer, "_device", "cpu")
        feats = []
        for syn, _ in choices:
            feats.append(torch.tensor([float(getattr(syn, "weight", 0.0))], device=device))
        logits = torch.stack([self.actor_w * f for f in feats]).squeeze(-1)
        probs = torch.softmax(logits, dim=0)
        idx = int(torch.multinomial(probs, 1).item())
        self._last_logprob = torch.log(probs[idx] + 1e-9)
        return choices[idx]

    def on_step(
        self,
        wanderer,
        current,
        next_syn,
        direction,
        step_index,
        out_value: Any,
    ) -> None:
        device = getattr(wanderer, "_device", "cpu")
        wL, wB, wD, gamma = self._params(wanderer)
        loss_t = wanderer._walk_ctx.get("cur_loss_tensor")
        if loss_t is None:
            loss_t = torch.tensor(0.0, device=device)
        beta_t = wanderer._walk_ctx.get("beta_t", torch.tensor(0.0, device=device))
        dist_t = wanderer._walk_ctx.get("action_distance", torch.tensor(0.0, device=device))
        r_t = -wL * loss_t + wB * beta_t + wD * dist_t
        # Forward-return logic: `_ret` caches the return from the next state.
        # Keep a copy before computing the current return so that after the
        # update we can push ``G`` forward for the following step.
        next_ret = self._ret
        G = r_t + gamma * next_ret
        value = self.critic_w * loss_t
        advantage = G - value
        actor_loss = -self._last_logprob * advantage.detach()
        critic_loss = advantage.pow(2)
        total = actor_loss + critic_loss
        total.backward(retain_graph=True)
        with torch.no_grad():
            self.actor_w -= self.lr * self.actor_w.grad
            self.actor_w.grad.zero_()
            self.critic_w -= self.lr * self.critic_w.grad
            self.critic_w.grad.zero_()
            for p in (wL, wB, wD, gamma):
                if getattr(p, "grad", None) is not None:
                    p -= self.lr * p.grad
                    p.grad.zero_()
        self._ret = G.detach()


__all__ = ["ActorCriticPlugin"]

PLUGIN_NAME = "actorcritic"
