from __future__ import annotations

from typing import Any
import torch




class MixedPrecisionPlugin:
    """Enable mixed precision training for Wanderer walks."""

    def before_walk(self, wanderer: "Wanderer", start: "Neuron") -> None:  # noqa: D401
        setattr(wanderer, "_use_mixed_precision", True)
        if not hasattr(wanderer, "_amp_scaler") or getattr(wanderer, "_amp_scaler", None) is None:
            try:
                scaler = torch.amp.GradScaler(
                    device_type="cuda", enabled=torch.cuda.is_available()
                )
            except Exception:
                scaler = None
            setattr(wanderer, "_amp_scaler", scaler)


__all__ = ["MixedPrecisionPlugin"]
