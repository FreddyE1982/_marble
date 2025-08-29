import unittest


class TestAutoPluginExplicit(unittest.TestCase):
    def test_explicit_plugin_not_deactivated(self):
        from marble.marblemain import Brain, Wanderer
        from marble.plugins.wanderer_autoplugin import AutoPlugin
        from marble.plugins.wanderer_epsgreedy import EpsilonGreedyChooserPlugin

        b = Brain(1, size=(1,))
        idx = b.available_indices()[0]
        b.add_neuron(idx, tensor=[1.0], type_name="autoneuron")
        w = Wanderer(
            b,
            type_name="epsilongreedy,autoplugin",
            neuroplasticity_type="base",
            seed=0,
        )
        w.ensure_learnable_param("autoplugin_bias_EpsilonGreedyChooserPlugin", -10.0)
        auto = next(p for p in w._wplugins if isinstance(p, AutoPlugin))
        self.assertTrue(auto.is_active(w, "EpsilonGreedyChooserPlugin", None))
        self.assertTrue(
            any(isinstance(p, EpsilonGreedyChooserPlugin) for p in w._wplugins)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

