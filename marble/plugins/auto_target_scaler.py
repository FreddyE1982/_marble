from __future__ import annotations

import torch
from typing import Any, List


class AutoTargetScalerPlugin:
    """Scale targets to roughly match output magnitude.

    During the first ``observe_steps`` calls it records the mean and standard
    deviation of model outputs and their corresponding targets. Once enough
    samples are collected, it computes a scale factor ``std_out / std_tgt`` and
    wraps the Wanderer's target provider so all future targets are multiplied by
    this factor. This helps to avoid exploding losses when targets are orders of
    magnitude smaller than outputs.
    """

    def __init__(self, observe_steps: int = 5) -> None:
        self.observe_steps = int(observe_steps)
        self._orig_tp = None
        self._seen = 0
        self._out_vals: List[torch.Tensor] = []
        self._tgt_vals: List[torch.Tensor] = []
        self.scale = 1.0

    def on_init(self, wanderer: "Wanderer") -> None:  # noqa: D401
        self._orig_tp = getattr(wanderer, "_target_provider", None)

    def loss(self, wanderer: "Wanderer", outputs: List[Any]):  # noqa: D401
        torch_mod = getattr(wanderer, "_torch", torch)
        device = getattr(wanderer, "_device", "cpu")
        if self._seen < self.observe_steps and outputs and self._orig_tp:
            y = outputs[-1]
            tgt = self._orig_tp(y)
            if not hasattr(tgt, "to"):
                tgt = torch_mod.tensor(tgt, dtype=torch_mod.float32, device=device)
            y = y.float()
            tgt = tgt.float().view_as(y)
            self._out_vals.append(y.detach().to("cpu"))
            self._tgt_vals.append(tgt.detach().to("cpu"))
            self._seen += 1
            if self._seen >= self.observe_steps:
                out_tensor = torch.stack(self._out_vals).view(-1)
                tgt_tensor = torch.stack(self._tgt_vals).view(-1)
                mean_out = out_tensor.mean()
                mean_tgt = tgt_tensor.mean()
                std_out = out_tensor.std()
                std_tgt = tgt_tensor.std()
                self.mean_out = mean_out
                self.mean_tgt = mean_tgt
                self.std_out = std_out
                self.std_tgt = std_tgt
                if std_tgt > 0 and std_out > 0:
                    self.scale = float((std_out / std_tgt).item())
                elif mean_tgt != 0:
                    self.scale = float((mean_out / mean_tgt).item())
                else:
                    self.scale = 1.0
                orig_tp = self._orig_tp
                scale = self.scale

                def scaled_tp(_y: Any) -> Any:
                    t = orig_tp(_y)
                    return t * scale

                wanderer._target_provider = scaled_tp
                self._out_vals.clear()
                self._tgt_vals.clear()
        return torch_mod.tensor(0.0, device=device)


__all__ = ["AutoTargetScalerPlugin"]
