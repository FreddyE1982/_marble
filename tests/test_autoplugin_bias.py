import unittest


class TestAutoPluginBias(unittest.TestCase):
    def test_bias_can_deactivate_plugin(self):
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
        self.assertFalse(active)
        self.assertIn("EpsilonGreedyChooserPlugin", stack)

    def test_plugin_learnables_force_activation(self):
        from marble.marblemain import Brain, Wanderer
        from marble.plugins.wanderer_autoplugin import AutoPlugin
        import torch

        b = Brain(1, size=(1,))
        idx = b.available_indices()[0]
        b.add_neuron(idx, tensor=[1.0], type_name="autoneuron")
        w = Wanderer(
            b,
            type_name="autoplugin,actorcritic",
            neuroplasticity_type="base",
            seed=0,
        )
        auto = next(p for p in w._wplugins if isinstance(p, AutoPlugin))
        bias_name = "autoplugin_bias_ActorCriticPlugin"
        bias = w.get_learnable_param_tensor(bias_name)
        bias.data.fill_(-100.0)

        original_sample = torch.distributions.MultivariateNormal.sample

        def fake_sample(self, sample_shape=torch.Size()):
            return torch.zeros_like(self.loc)

        torch.distributions.MultivariateNormal.sample = fake_sample  # type: ignore[assignment]
        try:
            w.set_param_optimization(bias_name, enabled=False)
            auto._gate_cache.clear()
            active_before = auto.is_active(w, "ActorCriticPlugin", None)
            self.assertFalse(active_before)

            w.set_param_optimization(bias_name, enabled=True)
            auto._gate_cache.clear()
            active_after_bias = auto.is_active(w, "ActorCriticPlugin", None)
            self.assertTrue(active_after_bias)

            # Disable bias optimisation again and enable a plugin-specific learnable.
            w.set_param_optimization(bias_name, enabled=False)
            actor = next(
                getattr(p, "_plugin", p)
                for p in w._wplugins
                if getattr(getattr(p, "_plugin", p), "__class__").__name__ == "ActorCriticPlugin"
            )
            actor._params(w)  # ensure learnables are registered
            w.set_param_optimization("w_loss", enabled=True)
            bias.data.fill_(-100.0)
            auto._gate_cache.clear()
            active_after_param = auto.is_active(w, "ActorCriticPlugin", None)
            self.assertTrue(active_after_param)
        finally:
            torch.distributions.MultivariateNormal.sample = original_sample  # type: ignore[assignment]


if __name__ == "__main__":
    unittest.main(verbosity=2)
