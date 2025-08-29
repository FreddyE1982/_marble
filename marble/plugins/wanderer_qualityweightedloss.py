from __future__ import annotations

import torch
from typing import Any, List




class QualityWeightedLossPlugin:
    """Scale loss by target quality so higher-quality samples influence updates more."""

    def loss(self, wanderer: "Wanderer", outputs: List[Any]):  # noqa: D401
        torch_device = getattr(wanderer, "_device", "cpu")
        target_provider = getattr(wanderer, "_target_provider", None)
        if not outputs or target_provider is None:
            return torch.tensor(0.0, device=torch_device)
        y = outputs[-1].float()
        tgt = target_provider(outputs[-1])  # type: ignore[call-arg]
        if not hasattr(tgt, "float"):
            tgt = torch.tensor(tgt, dtype=torch.float32, device=torch_device)
        tgt = tgt.view_as(y)
        weight = tgt.clamp(min=0.0)
        return (weight * (y - tgt) ** 2).mean()

__all__ = ["QualityWeightedLossPlugin"]

