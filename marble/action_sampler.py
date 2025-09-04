from __future__ import annotations

"""Utility functions for sampling plugin actions based on embeddings.

The module exposes helpers to compute logits from plugin embeddings and
sample a subset of plugin identifiers via either Bernoulli sampling or
Gumbel top-k selection."""

from typing import List

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


def sample_actions(
    logits: torch.Tensor,
    *,
    mode: str = "bernoulli",
    top_k: int = 1,
) -> torch.Tensor:
    """Sample plugin actions from ``logits``.

    Parameters
    ----------
    logits:
        Tensor of shape ``(num_plugins,)`` with unnormalised log probabilities.
    mode:
        Either ``"bernoulli"`` for independent Bernoulli trials or
        ``"gumbel-top-k"`` for Gumbel top-k sampling.
    top_k:
        Number of selections when ``mode`` is ``"gumbel-top-k"``.
    """
    if mode == "bernoulli":
        probs = torch.sigmoid(logits)
        return torch.bernoulli(probs).to(torch.long)
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
) -> List[int]:
    """Return a subset of ``plugin_ids`` based on sampled actions."""
    logits = compute_logits(e_t, e_a_t)
    mask = sample_actions(logits, mode=mode, top_k=top_k)
    indices = mask.nonzero(as_tuple=False).squeeze(1)
    return plugin_ids[indices].tolist()


__all__ = ["compute_logits", "sample_actions", "select_plugins"]
