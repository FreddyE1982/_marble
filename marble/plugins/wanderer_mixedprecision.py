from __future__ import annotations

from typing import Any
import torch


class MixedPrecisionPlugin:
    """Enable mixed precision training for Wanderer walks."""

    def before_walk(self, wanderer: "Wanderer", start: "Neuron") -> None:  # noqa: D401
        # Only enable mixed precision on CUDA devices to avoid CPU precision issues
        if not torch.cuda.is_available() or not str(getattr(wanderer, "_device", "cpu")).startswith("cuda"):
            return
        setattr(wanderer, "_use_mixed_precision", True)
        if not hasattr(wanderer, "_amp_scaler") or getattr(wanderer, "_amp_scaler", None) is None:
            try:
                scaler = torch.amp.GradScaler(device_type="cuda", enabled=True)
            except Exception:
                scaler = None
            setattr(wanderer, "_amp_scaler", scaler)


__all__ = ["MixedPrecisionPlugin"]
