from __future__ import annotations

"""Encoding utilities for plugin-driven decision networks.

This module provides the :class:`PluginEncoder` which maps plugin IDs to
learnable embeddings and combines them with encoded walker context and past
actions.  It is designed so that every plugin automatically receives a stable
``plugin_id`` during registration.  Future plugins registered through the
automatic discovery mechanism in :mod:`marble.plugins` will therefore be
compatible without additional boilerplate.
"""

from typing import Iterable

import torch
from torch import nn


class PluginEncoder(nn.Module):
    """Encode plugin identifiers alongside context and action history.

    Parameters
    ----------
    num_plugins:
        Total number of registered plugins.  This determines the vocabulary
        size for the embedding table.  The value can be obtained from
        ``len(marble.plugins.PLUGIN_ID_REGISTRY)``.
    embed_dim:
        Dimensionality of the embedding vectors for plugin IDs.
    action_dim:
        Dimensionality used for summarising past actions.
    ctx_dim:
        Size of each context vector fed into the recurrent unit.
    rnn_type:
        Either ``"gru"`` or ``"lstm"`` specifying which recurrent unit is
        used to aggregate the walker context.
    """

    def __init__(
        self,
        num_plugins: int,
        embed_dim: int = 16,
        action_dim: int = 16,
        ctx_dim: int = 16,
        *,
        rnn_type: str = "gru",
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_plugins, embed_dim)
        rnn_cls = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        self.ctx_rnn = rnn_cls(ctx_dim, ctx_dim, batch_first=True)
        # Reuse the plugin embedding matrix for past actions so that action
        # identifiers correspond to plugin IDs.
        self.action_embed = self.embedding

    def forward(
        self,
        plugin_ids: torch.Tensor,
        ctx_seq: torch.Tensor,
        past_action_ids: torch.Tensor | Iterable[int],
    ) -> torch.Tensor:
        """Return concatenated feature representation.

        Parameters
        ----------
        plugin_ids:
            Tensor of shape ``(batch,)`` containing the IDs of plugins for
            which features are being generated.
        ctx_seq:
            Context tensor of shape ``(batch, steps, ctx_dim)`` describing the
            walker state across ``steps`` time steps.
        past_action_ids:
            Tensor of shape ``(batch, history)`` listing previously executed
            plugin IDs.  The embedding of these IDs is averaged to produce a
            fixed-size vector representing the action history.
        """

        if not isinstance(past_action_ids, torch.Tensor):
            past_action_ids = torch.tensor(list(past_action_ids), dtype=torch.long, device=plugin_ids.device)
            past_action_ids = past_action_ids.unsqueeze(0).expand(plugin_ids.size(0), -1)

        plug_emb = self.embedding(plugin_ids)
        _, h_ctx = self.ctx_rnn(ctx_seq)
        if isinstance(h_ctx, tuple):  # LSTM returns (h, c)
            h_ctx = h_ctx[0]
        h_ctx = h_ctx.squeeze(0)
        past_emb = self.action_embed(past_action_ids)
        past_emb = past_emb.mean(dim=1)
        return torch.cat([plug_emb, h_ctx, past_emb], dim=-1)


__all__ = ["PluginEncoder"]
