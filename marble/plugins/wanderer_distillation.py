from __future__ import annotations

from typing import Any, Optional, List

from ..wanderer import register_wanderer_type


class DistillationPlugin:
    """Adds a simple distillation loss to match a moving-average 'teacher' of outputs.

    Config via wanderer._neuro_cfg:
      - distill_lambda (default 0.1)
      - teacher_momentum (EMA, default 0.9)
    """

    def __init__(self) -> None:
        self._ema: Optional[Any] = None

    def loss(self, wanderer: "Wanderer", outputs: List[Any]):  # noqa: D401
        torch = getattr(wanderer, "_torch", None)
        if torch is None or not outputs:
            return 0.0 if torch is None else torch.tensor(0.0, device=getattr(wanderer, "_device", "cpu"))
        dev = getattr(wanderer, "_device", "cpu")
        lam = float(getattr(wanderer, "_neuro_cfg", {}).get("distill_lambda", 0.1))
        mom = float(getattr(wanderer, "_neuro_cfg", {}).get("teacher_momentum", 0.9))
        # Use the last output for current prediction
        y = outputs[-1]
        if not hasattr(y, "detach"):
            y = torch.tensor([float(v) for v in (y if isinstance(y, (list, tuple)) else [y])], dtype=torch.float32, device=dev)
        yt = y.detach().to(dev).float().view(-1)
        if self._ema is None:
            self._ema = yt.detach()
        else:
            self._ema = (mom * self._ema) + ((1.0 - mom) * yt.detach())
        # MSE between current y and teacher ema
        if self._ema.shape != yt.shape:
            # Align to min length
            m = min(int(self._ema.numel()), int(yt.numel()))
            ema = self._ema.view(-1)[:m]
            yv = yt.view(-1)[:m]
        else:
            ema = self._ema
            yv = yt
        return lam * ((yv - ema).pow(2).mean())


try:
    register_wanderer_type("distillation", DistillationPlugin())
except Exception:
    pass

__all__ = ["DistillationPlugin"]
