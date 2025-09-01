import os
import sys
import types
import importlib
import unittest
from unittest.mock import patch

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from marble.plugins.wanderer_resource_allocator import ResourceAllocatorPlugin


class ResourceAllocatorOOMTests(unittest.TestCase):
    def test_cuda_oom_triggers_disk_offload(self):
        plug = ResourceAllocatorPlugin()
        obj = types.SimpleNamespace()
        obj.weight = torch.ones(4)

        module = importlib.import_module("marble.plugins.wanderer_resource_allocator")
        module.psutil = types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(available=0))

        orig_to = torch.Tensor.to

        def fake_to(self, *args, **kwargs):
            device = args[0] if args else kwargs.get("device")
            if device == "cuda":
                raise torch.cuda.OutOfMemoryError("CUDA OOM")
            return orig_to(self, *args, **kwargs)

        with patch.object(torch.Tensor, "to", fake_to):
            plug._safe_transfer(obj, "weight", obj.weight, "cuda")

        off_path = getattr(obj, "_weight_offload")
        print("offload path:", off_path)
        self.assertTrue(isinstance(off_path, str) and os.path.exists(off_path))
        self.assertEqual(obj.weight.numel(), 0)

        with patch.object(torch.Tensor, "to", orig_to):
            plug._safe_transfer(obj, "weight", obj.weight, "cpu")

        print("reloaded size:", obj.weight.numel())
        self.assertGreater(obj.weight.numel(), 0)
        self.assertIsNone(getattr(obj, "_weight_offload"))


if __name__ == "__main__":
    unittest.main()
