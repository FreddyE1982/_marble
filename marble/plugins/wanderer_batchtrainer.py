from __future__ import annotations

from typing import Any


class BatchTrainingPlugin:
    """Configure Wanderer to operate on batches of samples at once.

    The desired batch size can be provided via the Wanderer's neuro_config
    using the key ``batch_size``. When attached, ``run_training_with_datapairs``
    will combine datapairs into batches and feed them through the Wanderer in
    one go, letting each step process all samples simultaneously.
    """

    def on_init(self, wanderer: "Wanderer") -> None:  # noqa: D401
        cfg = getattr(wanderer, "_neuro_cfg", {}) or {}
        bs = int(cfg.get("batch_size", 1))
        setattr(wanderer, "_batch_size", bs)

    def loss(self, wanderer: "Wanderer", outputs: list[Any]):  # noqa: D401
        torch = getattr(wanderer, "_torch", None)
        if torch is None or not outputs:
            return 0.0 if torch is None else torch.tensor(0.0, device=getattr(wanderer, "_device", "cpu"))
        yt = outputs[-1].float()
        tgt = wanderer._target_provider(outputs[-1])  # type: ignore[attr-defined]
        if not hasattr(tgt, "float"):
            tgt = torch.tensor(tgt, dtype=torch.float32, device=getattr(wanderer, "_device", "cpu"))
        tgt = tgt.view_as(yt)
        return torch.nn.functional.mse_loss(yt, tgt)


__all__ = ["BatchTrainingPlugin"]
