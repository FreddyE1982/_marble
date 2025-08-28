from __future__ import annotations

from typing import Any
import torch

from ..wanderer import register_wanderer_type


class MixedPrecisionPlugin:
    """Enable mixed precision training for Wanderer walks."""

    def before_walk(self, wanderer: "Wanderer", start: "Neuron") -> None:  # noqa: D401
        if not torch.cuda.is_available():
            setattr(wanderer, "_use_mixed_precision", False)
            setattr(wanderer, "_amp_scaler", None)
            return
        setattr(wanderer, "_use_mixed_precision", True)
        if not hasattr(wanderer, "_amp_scaler") or getattr(wanderer, "_amp_scaler", None) is None:
            try:
                scaler = torch.amp.GradScaler("cuda")
            except Exception:
                scaler = None
            setattr(wanderer, "_amp_scaler", scaler)


try:
    register_wanderer_type("mixedprecision", MixedPrecisionPlugin())
except Exception:
    pass

__all__ = ["MixedPrecisionPlugin"]
