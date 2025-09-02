import unittest
from unittest.mock import patch
import types

import torch

from marble.plugins.wanderer_resource_allocator import (
    ResourceAllocatorPlugin,
    TENSOR_REGISTRY,
)


class ResourceAllocatorDecayTests(unittest.TestCase):
    def test_hits_decay_over_time(self) -> None:
        plug = ResourceAllocatorPlugin()

        class Holder:
            pass

        obj = Holder()
        obj.weight = torch.ones(10)
        TENSOR_REGISTRY.register(obj, "weight")

        class DummyW:
            def __init__(self) -> None:
                self._plugin_state = {
                    "resource_hits": {},
                    "base_score": 0.0,
                    "last_time": 0.0,
                }
                self._learnables = {}

            def ensure_learnable_param(self, name, init):
                self._learnables[name] = torch.tensor(init)

            def get_learnable_param_tensor(self, name):
                return self._learnables[name]

            def _compute_loss(self, outputs):
                return torch.tensor(0.0)

            _walk_ctx = types.SimpleNamespace(outputs=[])

        w = DummyW()

        with patch("torch.cuda.is_available", return_value=False), patch(
            "time.perf_counter", side_effect=[0.0, 0.0, 100.0, 100.0]
        ):
            plug.rebalance_all(w)
            plug.rebalance_all(w)

        hits = obj._tensor_hits["weight"][0]
        print("decayed_hits", hits)
        self.assertLess(hits, 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

