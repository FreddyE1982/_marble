import unittest


class TestAutoPluginBias(unittest.TestCase):
    def test_bias_does_not_deactivate_wplugin(self):
        from marble.marblemain import Brain, Wanderer
        from marble.plugins.wanderer_autoplugin import AutoPlugin

        b = Brain(1, size=(1,))
        idx = b.available_indices()[0]
        b.add_neuron(idx, tensor=[1.0], type_name="autoneuron")
        w = Wanderer(
            b,
            type_name="epsilongreedy,autoplugin",
            neuroplasticity_type="base",
            seed=0,
        )
        w.ensure_learnable_param("autoplugin_bias_EpsilonGreedyChooserPlugin", 0.0)
        w.get_learnable_param_tensor("autoplugin_bias_EpsilonGreedyChooserPlugin").data.fill_(-10.0)
        auto = next(p for p in w._wplugins if isinstance(p, AutoPlugin))
        active = auto.is_active(w, "EpsilonGreedyChooserPlugin", None)
        stack = [getattr(getattr(p, "_plugin", p), "__class__").__name__ for p in w._wplugins]
        print("plugin active:", active)
        print("stack:", stack)
        self.assertTrue(active)
        self.assertIn("EpsilonGreedyChooserPlugin", stack)


if __name__ == "__main__":
    unittest.main(verbosity=2)
