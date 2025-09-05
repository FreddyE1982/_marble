import unittest
import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class AutoPluginDeviceTest(unittest.TestCase):
    def test_is_active_cuda(self) -> None:
        from marble.marblemain import Brain, Wanderer
        from marble.plugins.wanderer_autoplugin import AutoPlugin

        b = Brain(1, size=(1,))
        idx = b.available_indices()[0]
        b.add_neuron(idx, tensor=[1.0], type_name="autoneuron")
        w = Wanderer(b, type_name="autoplugin", neuroplasticity_type="base", seed=0)
        w._device = torch.device("cuda")
        w.ensure_learnable_param("autoplugin_bias_dummy", 0.0)
        w.ensure_learnable_param("autoplugin_gain_dummy", 1.0)
        plugin = next(p for p in w._wplugins if isinstance(p, AutoPlugin))
        self.assertIsInstance(plugin.is_active(w, "dummy", None), bool)


if __name__ == "__main__":
    unittest.main(verbosity=2)
