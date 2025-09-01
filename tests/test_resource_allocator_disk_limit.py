import os
import sys
import types
import unittest
from unittest.mock import patch

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from marble.plugins.wanderer_resource_allocator import ResourceAllocatorPlugin


class ResourceAllocatorDiskLimitTests(unittest.TestCase):
    def test_disk_limit_blocks_offload(self) -> None:
        plug = ResourceAllocatorPlugin()
        plug.max_disk_mb = 0.0
        obj = types.SimpleNamespace()
        obj.weight = torch.ones(4)

        module = __import__("marble.plugins.wanderer_resource_allocator", fromlist=['psutil'])
        module.psutil = types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(available=0))

        orig_to = torch.Tensor.to

        def fake_to(self, *args, **kwargs):
            device = args[0] if args else kwargs.get("device")
            if device == "cuda":
                raise torch.cuda.OutOfMemoryError("CUDA OOM")
            return orig_to(self, *args, **kwargs)

        with patch.object(torch.Tensor, "to", fake_to):
            plug._safe_transfer(obj, "weight", obj.weight, "cuda")

        self.assertFalse(isinstance(getattr(obj, "_weight_offload", None), str))
        self.assertEqual(obj.weight.numel(), 0)


if __name__ == "__main__":
    unittest.main()
