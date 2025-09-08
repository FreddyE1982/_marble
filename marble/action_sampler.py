from __future__ import annotations

"""Utility functions for sampling plugin actions based on embeddings.

The module exposes helpers to compute logits from plugin embeddings and
sample a subset of plugin identifiers via either Bernoulli sampling or
Gumbel top-k selection."""

from typing import Dict, Iterable, List, Set

import torch


def compute_logits(e_t: torch.Tensor, e_a_t: torch.Tensor) -> torch.Tensor:
    """Return logits ``z_i`` from plugin and action embeddings.

    Parameters
    ----------
    e_t:
        Tensor of shape ``(num_plugins, d)`` containing embeddings for each
        plugin.
    e_a_t:
        Tensor of shape ``(d,)`` representing the aggregated action
        embedding.
    """
    return e_t @ e_a_t


def _project_constraints(
    relaxed: torch.Tensor,
    *,
    costs: torch.Tensor | None = None,
    budget: float = float("inf"),
    incompatibility: Dict[int, Set[int]] | None = None,
) -> torch.Tensor:
    """Project ``relaxed`` activations onto the constraint set ``𝔄``.

    The function applies a straight-through estimator so gradients flow through
    ``relaxed`` while the forward pass respects hard budget and
    incompatibility constraints.
    """

    hard = (relaxed > 0.5).to(relaxed.dtype)
    if incompatibility:
        for i, others in incompatibility.items():
            for j in others:
                if hard[i] > 0 and hard[j] > 0:
                    if relaxed[i] >= relaxed[j]:
                        hard[j] = 0.0
                    else:
                        hard[i] = 0.0
    if costs is not None and budget < float("inf"):
        total = (hard * costs).sum()
        if total > budget:
            order = torch.argsort(relaxed, descending=True)
            new_hard = torch.zeros_like(hard)
            spent = torch.tensor(0.0, device=relaxed.device)
            for idx in order:
                c = costs[idx]
                if hard[idx] > 0 and spent + c <= budget:
                    new_hard[idx] = hard[idx]
                    spent = spent + c
            hard = new_hard
    return hard + (relaxed - relaxed.detach())


def sample_actions(
    logits: torch.Tensor,
    *,
    mode: str = "bernoulli",
    top_k: int = 1,
    temperature: float = 1.0,
    costs: torch.Tensor | None = None,
    budget: float = float("inf"),
    incompatibility: Dict[int, Set[int]] | None = None,
) -> torch.Tensor:
    """Sample plugin actions from ``logits``.

    Parameters
    ----------
    logits:
        Tensor of shape ``(num_plugins,)`` with unnormalised log probabilities.
    mode:
        ``"bernoulli"`` for independent Bernoulli trials, ``"bernoulli-relaxed``
        for a Concrete distribution with straight-through projection or
        ``"gumbel-top-k"`` for Gumbel top-k sampling.
    top_k:
        Number of selections when ``mode`` is ``"gumbel-top-k"``.
    temperature:
        Relaxation temperature when ``mode`` is ``"bernoulli-relaxed"``.
    costs, budget, incompatibility:
        Constraint parameters for ``"bernoulli-relaxed"`` mode.
    """
    if mode == "bernoulli":
        probs = torch.sigmoid(logits)
        return torch.bernoulli(probs).to(torch.long)
    if mode == "bernoulli-relaxed":
        gumbels = -torch.log(-torch.log(torch.rand_like(logits)))
        relaxed = torch.sigmoid((logits + gumbels) / max(temperature, 1e-8))
        return _project_constraints(
            relaxed,
            costs=costs,
            budget=budget,
            incompatibility=incompatibility,
        )
    if mode == "gumbel-top-k":
        gumbels = -torch.log(-torch.log(torch.rand_like(logits)))
        scores = logits + gumbels
        _, idx = scores.topk(top_k)
        out = torch.zeros_like(logits, dtype=torch.long)
        out.scatter_(0, idx, 1)
        return out
    raise ValueError(f"unknown sampling mode: {mode}")


def select_plugins(
    plugin_ids: torch.Tensor,
    e_t: torch.Tensor,
    e_a_t: torch.Tensor,
    *,
    mode: str = "bernoulli",
    top_k: int = 1,
    **kwargs,
) -> List[int]:
    """Return a subset of ``plugin_ids`` based on sampled actions."""
    logits = compute_logits(e_t, e_a_t)
    mask = sample_actions(logits, mode=mode, top_k=top_k, **kwargs)
    indices = (mask > 0.5).nonzero(as_tuple=False).squeeze(1)
    return plugin_ids[indices].tolist()


__all__ = ["compute_logits", "sample_actions", "select_plugins"]
