import unittest


class TestAutoPluginExclude(unittest.TestCase):
    def test_excluded_plugin_removed(self):
        from marble.marblemain import Brain, Wanderer, register_wanderer_type
        from marble.plugins.wanderer_autoplugin import AutoPlugin

        b = Brain(1, size=(1,))
        idx = b.available_indices()[0]
        b.add_neuron(idx, tensor=[1.0], type_name="autoneuron")

        # Register AutoPlugin variant that disables EpsilonGreedyChooserPlugin
        register_wanderer_type(
            "autoplugin_no_eps",
            AutoPlugin(disabled_plugins=["EpsilonGreedyChooserPlugin"]),
        )

        w = Wanderer(b, type_name="epsilongreedy,autoplugin_no_eps", neuroplasticity_type="base", seed=0)

        # The excluded plugin should not be present in the wanderer stack
        plugins = [p.__class__.__name__ for p in w._wplugins]
        print("plugins:", plugins)
        self.assertFalse(
            any("EpsilonGreedyChooserPlugin" == name for name in plugins)
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)

