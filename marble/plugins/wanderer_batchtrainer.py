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
        """No additional loss; base Wanderer loss already handles batches."""
        torch = getattr(wanderer, "_torch", None)
        return 0.0 if torch is None else torch.tensor(0.0, device=getattr(wanderer, "_device", "cpu"))


__all__ = ["BatchTrainingPlugin"]
