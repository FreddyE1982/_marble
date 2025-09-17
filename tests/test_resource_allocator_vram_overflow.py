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

    def test_handle_cuda_oom_moves_low_hit_tensors(self):
        plug = ResourceAllocatorPlugin()
        dummy = types.SimpleNamespace()
        dummy.weight = torch.ones(8)
        module = importlib.import_module("marble.plugins.wanderer_resource_allocator")

        base_tensor = torch.ones(8)

        class FakeCudaTensor:
            def __init__(self, base):
                self._base = base

            def __getattr__(self, item):
                return getattr(self._base, item)

            @property
            def device(self):
                class _FakeDevice:
                    def __str__(self_inner):
                        return "cuda:0"

                    def lower(self_inner):
                        return "cuda:0"

                return _FakeDevice()

        fake_tensor = FakeCudaTensor(base_tensor)
        dummy.weight = fake_tensor
        fake_entries = [(dummy, "weight", fake_tensor, 0.05)]
        transfers: list[tuple[str, str]] = []

        def fake_transfer(obj, attr, tensor, target):
            transfers.append((attr, target))

        orig_is_tensor = torch.is_tensor

        def fake_is_tensor(obj):
            if isinstance(obj, FakeCudaTensor):
                return True
            return orig_is_tensor(obj)

        with patch.object(module.TENSOR_REGISTRY, "iter_tensors", return_value=fake_entries):
            with patch("marble.plugins.wanderer_resource_allocator.torch.cuda.is_available", return_value=True):
                with patch("marble.plugins.wanderer_resource_allocator.torch.cuda.empty_cache"):
                    with patch.object(ResourceAllocatorPlugin, "_system_metrics", return_value={"gpu": 1.0, "vram": 0.0, "cpu": 0.0, "ram": 0.2, "disk": 0.1}):
                        with patch("marble.plugins.wanderer_resource_allocator.torch.is_tensor", new=fake_is_tensor):
                            with patch.object(ResourceAllocatorPlugin, "_safe_transfer", side_effect=fake_transfer):
                                handled = plug.handle_cuda_oom(None, torch.cuda.OutOfMemoryError("Tried to allocate 64.00 MiB"))

        print("oom handled transfers:", transfers)
        self.assertTrue(handled)
        self.assertGreater(len(transfers), 0)
        self.assertEqual(transfers[0][1], "cpu")


if __name__ == "__main__":
    unittest.main()
