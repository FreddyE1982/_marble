import types
import unittest
from unittest.mock import patch

import torch

from marble.plugins.wanderer_resource_allocator import (
    ResourceAllocatorPlugin,
    TENSOR_REGISTRY,
)


class ResourceAllocatorSmallTensorTests(unittest.TestCase):
    def test_small_tensor_stays_on_cpu(self) -> None:
        with patch(
            "marble.plugins.wanderer_resource_allocator._load_resource_cfg",
            return_value={"min_gpu_tensor_mb": 1.0},
        ):
            plug = ResourceAllocatorPlugin()
        class Holder:
            pass

        obj = Holder()
        obj.weight = torch.ones(10)
        TENSOR_REGISTRY.register(obj, "weight")

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
        with patch("torch.cuda.is_available", return_value=True):
            plug.rebalance_all(w)
        print("device", obj.weight.device.type)
        self.assertEqual(obj.weight.device.type, "cpu")


if __name__ == "__main__":
    unittest.main(verbosity=2)
