import os
import types
import importlib
import unittest
from unittest.mock import patch

import torch

from marble.plugins.wanderer_resource_allocator import (
    ResourceAllocatorPlugin,
    TENSOR_REGISTRY,
)


class ResourceAllocatorRamPressureTests(unittest.TestCase):
    def test_ram_pressure_offloads_to_disk(self) -> None:
        plug = ResourceAllocatorPlugin()
        plug.ram_offload_threshold = 0.0
        class Holder:
            pass

        obj = Holder()
        obj.weight = torch.ones(10_000_000)
        TENSOR_REGISTRY.register(obj, "weight")

        module = importlib.import_module("marble.plugins.wanderer_resource_allocator")
        module.psutil = types.SimpleNamespace(
            virtual_memory=lambda: types.SimpleNamespace(percent=99, available=10**9),
            disk_usage=lambda _path: types.SimpleNamespace(percent=0),
            cpu_percent=lambda: 0,
        )

        class DummyW:
            def __init__(self):
                self._plugin_state = {"resource_hits": {}}
                self._learnables = {}

            def ensure_learnable_param(self, name, init):
                self._learnables[name] = torch.tensor(init)

            def get_learnable_param_tensor(self, name):
                return self._learnables[name]

            def _compute_loss(self, outputs):
                return torch.tensor(0.0)

            _walk_ctx = types.SimpleNamespace(outputs=[])

        w = DummyW()
        with patch("torch.cuda.is_available", return_value=False):
            plug.rebalance_all(w)
        off_path = getattr(obj, "_weight_offload")
        print("offload path", off_path)
        self.assertTrue(isinstance(off_path, str) and os.path.exists(off_path))
        self.assertEqual(obj.weight.numel(), 0)


if __name__ == "__main__":
    unittest.main()
