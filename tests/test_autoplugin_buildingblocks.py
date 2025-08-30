import unittest

import marble.plugins  # noqa: F401 ensure plugin discovery
from marble.marblemain import Brain, Wanderer
from marble.plugins.wanderer_autoplugin import AutoPlugin


class TestAutoPluginBuildingBlocks(unittest.TestCase):
    def test_buildingblock_application(self):
        brain = Brain(1, size=(1,))
        w = Wanderer(brain, type_name="autoplugin", neuroplasticity_type="base", seed=0)
        auto = next(p for p in w._wplugins if isinstance(p, AutoPlugin))
        w.ensure_learnable_param("autoplugin_bias_create_neuron", 0.0)
        w.ensure_learnable_param("autoplugin_bias_change_neuron_weight", 0.0)
        w.get_learnable_param_tensor("autoplugin_bias_create_neuron").data.fill_(10.0)
        w.get_learnable_param_tensor("autoplugin_bias_change_neuron_weight").data.fill_(10.0)
        auto.apply_buildingblock(w, "create_neuron", index=(0,), tensor=[0.0])
        auto.apply_buildingblock(w, "change_neuron_weight", index=(0,), weight=2.0)
        n = brain.get_neuron((0,))
        print("neuron weight", getattr(n, "weight", None))
        self.assertIsNotNone(n)
        self.assertEqual(getattr(n, "weight", None), 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
