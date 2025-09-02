from __future__ import annotations

from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marble.plugins.auto_target_scaler import AutoTargetScalerPlugin


def test_auto_target_scaler() -> None:
    class DummyWanderer:
        def __init__(self) -> None:
            self._torch = torch
            self._device = "cpu"
            self._target_provider = lambda _y: torch.tensor(1e-6)

    w = DummyWanderer()
    plugin = AutoTargetScalerPlugin(observe_steps=2)
    plugin.on_init(w)
    out = torch.tensor(10.0)
    plugin.loss(w, [out])
    plugin.loss(w, [out])
    scaled_target = w._target_provider(out)
    loss_before = (out - torch.tensor(1e-6)) ** 2
    loss_after = (out - scaled_target) ** 2
    print("loss_before", float(loss_before), "loss_after", float(loss_after))
    assert float(loss_after) < float(loss_before)
