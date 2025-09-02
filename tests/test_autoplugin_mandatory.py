import unittest

import marble.plugins  # ensure plugin discovery
from marble.marblemain import Brain, Wanderer, register_wanderer_type
from marble.plugins.wanderer_autoplugin import AutoPlugin


class TestAutoPluginMandatory(unittest.TestCase):
    def test_mandatory_plugin_active(self):
        register_wanderer_type(
            "autoplugin_mand",
            AutoPlugin(mandatory_plugins=["EpsilonGreedyChooserPlugin"]),
        )
        b = Brain(1, size=(1,))
        idx = b.available_indices()[0]
        n = b.add_neuron(idx, tensor=[1.0], type_name="autoneuron")
        w = Wanderer(
            b, type_name="epsilongreedy,autoplugin_mand", neuroplasticity_type="base", seed=0
        )
        auto = next(p for p in w._wplugins if isinstance(p, AutoPlugin))
        w.ensure_learnable_param("autoplugin_bias_EpsilonGreedyChooserPlugin", 0.0)
        w.get_learnable_param_tensor("autoplugin_bias_EpsilonGreedyChooserPlugin").data.fill_(-10.0)
        active = auto.is_active(w, "EpsilonGreedyChooserPlugin", None)
        print("mandatory active:", active)
        self.assertTrue(active)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
