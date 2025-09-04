from __future__ import annotations

"""Recurrent encoder for decision history.

This module provides :class:`HistoryEncoder` which maintains a recurrent
summary of past interactions.  At each decision step the encoder receives the
current observation ``o_t``, the previous action vector ``a_{t-1}`` and the
previous reward ``r_{t-1}``.  These inputs are concatenated and processed by a
small GRU (or LSTM) to yield the new hidden state ``h_t`` which can be fed into
policy networks.
"""

from typing import Optional, Tuple

import torch
from torch import nn


class HistoryEncoder(nn.Module):
    """Encode ``(o_t, a_{t-1}, r_{t-1})`` into a hidden state ``h_t``.

    Parameters
    ----------
    obs_dim:
        Dimensionality of the observation vector ``o_t``.
    action_dim:
        Length of the one-hot action vector ``a_{t-1}``.
    hidden_dim:
        Size of the recurrent hidden state ``h_t``.
    use_lstm:
        If ``True`` an :class:`~torch.nn.LSTM` is used instead of a GRU.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        *,
        use_lstm: bool = False,
    ) -> None:
        super().__init__()
        rnn_cls = nn.LSTM if use_lstm else nn.GRU
        self.rnn = rnn_cls(obs_dim + action_dim + 1, hidden_dim, batch_first=True)

    def forward(
        self,
        o_t: torch.Tensor,
        a_prev: torch.Tensor,
        r_prev: torch.Tensor,
        h_prev: Optional[torch.Tensor | Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Return updated hidden state ``h_t``.

        ``o_t`` and ``a_prev`` are expected to be one-dimensional tensors while
        ``r_prev`` is a scalar tensor.  ``h_prev`` represents the previous hidden
        state and may be ``None`` for zero-initialisation.
        """

        x = torch.cat([o_t, a_prev, r_prev.view(1)], dim=-1)
        x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, input_dim)
        _, h_t = self.rnn(x, h_prev)
        return h_t


__all__ = ["HistoryEncoder"]
