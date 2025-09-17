import unittest

import torch

from marble.plugins.wanderer_resource_allocator import ResourceAllocatorPlugin


class _Holder:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.weight = tensor


class DiskGradientTests(unittest.TestCase):
    def test_disk_offload_preserves_gradients(self) -> None:
        plug = ResourceAllocatorPlugin()
        plug.max_disk_mb = 1024
        plug.compress_offload = False

        weight = torch.tensor([1.5, -2.0, 0.5], dtype=torch.float32, requires_grad=True)
        holder = _Holder(weight)

        original = holder.weight
        plug._safe_transfer(holder, "weight", holder.weight, "disk")

        path = getattr(holder, "_weight_offload", None)
        self.assertIsInstance(path, str)
        placeholder = holder.weight
        self.assertTrue(placeholder.requires_grad)
        self.assertEqual(placeholder.numel(), 0)

        loss = (original * 3.0).sum()
        loss.backward()

        meta = getattr(holder, "_weight_offmeta")
        self.assertIn("_grad_bridge", meta)
        self.assertTrue(torch.allclose(meta["_grad_bridge"]["value"], torch.tensor([3.0, 3.0, 3.0])))

        restored = plug.restore(holder, "weight", torch.device("cpu"))
        self.assertTrue(torch.allclose(restored, torch.tensor([1.5, -2.0, 0.5], dtype=restored.dtype)))
        self.assertIsNotNone(restored.grad)
        self.assertTrue(torch.allclose(restored.grad, torch.tensor([3.0, 3.0, 3.0], dtype=restored.grad.dtype)))

        plug.clear()
