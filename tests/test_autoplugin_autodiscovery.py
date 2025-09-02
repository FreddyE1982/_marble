import unittest

import marble.plugins  # ensure plugin discovery
from marble.marblemain import Brain, Wanderer, register_wanderer_type
from marble.plugins.wanderer_autoplugin import AutoPlugin


class TestAutoPluginAutoDiscovery(unittest.TestCase):
    def test_unlisted_plugin_not_added(self):
        register_wanderer_type("autoplugin_logger2", AutoPlugin())
        b = Brain(1, size=(1,))
        idx = b.available_indices()[0]
        n = b.add_neuron(idx, tensor=[1.0], type_name="autoneuron")
        w = Wanderer(b, type_name="autoplugin_logger2", neuroplasticity_type="base", seed=0)
        stack = [p.__class__.__name__ for p in w._wplugins]
        print("stack:", stack)
        self.assertNotIn("EpsilonGreedyChooserPlugin", stack)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
