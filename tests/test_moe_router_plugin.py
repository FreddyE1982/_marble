import unittest
from unittest import mock


class TestMoERouterPlugin(unittest.TestCase):
    def setUp(self) -> None:
        from marble.reporter import clear_report_group

        clear_report_group("decision_controller", "moe_router")
        clear_report_group("plugins", "moe_router")

    def test_router_registers_learnables_and_metrics(self) -> None:
        config = {
            "enabled": True,
            "decision_interval": 1,
            "capacity_factor": 1.5,
            "min_active_experts": 1,
            "max_active_experts": 4,
            "load_balance_alpha": 0.1,
            "load_balance_decay": 0.6,
            "budget_weight": 0.2,
        }
        with mock.patch("marble.plugins.wanderer_moe_router._load_moe_config", return_value=config):
            from marble.marblemain import Brain, Wanderer
            from marble.reporter import REPORTER

            brain = Brain(1, size=(1,))
            idx = brain.available_indices()[0]
            brain.add_neuron(idx, tensor=[1.0], type_name="autoneuron")

            wanderer = Wanderer(
                brain,
                type_name="moe_router,entropyaware",
                neuroplasticity_type="base",
                seed=0,
            )

            start = brain.neurons[idx]
            wanderer.walk(max_steps=1, start=start)

            bias = wanderer.get_learnable_param_tensor("moe_router_bias_entropyaware")
            gain = wanderer.get_learnable_param_tensor("moe_router_gain_entropyaware")
            self.assertIsNotNone(bias)
            self.assertIsNotNone(gain)

            stats = REPORTER.group("decision_controller", "moe_router", "scalars")
            self.assertIn("active_experts", stats)
            self.assertIn("budget_pressure", stats)

            per_plugin = REPORTER.group("plugins", "moe_router", "metrics")
            self.assertTrue(any(name in per_plugin for name in ("entropyaware", "resourceallocator")))

    def test_decision_controller_feedback_adjusts_parameters(self) -> None:
        from marble.decision_controller import DecisionController, PLUGIN_ID_REGISTRY
        from marble.plugin_encoder import PluginEncoder

        encoder = PluginEncoder(len(PLUGIN_ID_REGISTRY))
        controller = DecisionController(encoder=encoder)
        controller._moe_enabled = True
        controller._moe_cfg.update(
            {
                "enabled": True,
                "min_active_experts": 1,
                "max_active_experts": 5,
                "feedback_decay": 0.5,
                "cadence_scale": 0.5,
                "lambda_scale": 0.5,
                "topk_balance_scale": 0.5,
                "budget_target": 1.0,
            }
        )
        controller._moe_min_experts = 1
        controller._moe_max_experts = 5
        controller._moe_base_cadence = 2
        controller._moe_base_lambda = controller.lambda_lr
        controller._moe_base_top_k = max(1, controller.top_k)
        controller._moe_feedback = 0.0

        metrics = {
            "decision_controller/moe_router/scalars/active_experts": 3.0,
            "decision_controller/moe_router/scalars/budget_pressure": 2.0,
            "decision_controller/moe_router/scalars/load_balance": 1.5,
        }

        controller._apply_moe_feedback(metrics)

        self.assertGreaterEqual(controller.top_k, 3)
        self.assertGreater(controller.cadence, controller._moe_base_cadence - 1)
        self.assertGreater(controller.lambda_lr, controller._moe_base_lambda * 0.5)

    def test_router_disabled_keeps_all_plugins_active(self) -> None:
        config = {
            "enabled": False,
            "decision_interval": 1,
            "capacity_factor": 1.5,
            "min_active_experts": 1,
            "max_active_experts": 4,
        }
        with mock.patch(
            "marble.plugins.wanderer_moe_router._load_moe_config", return_value=config
        ):
            from marble.marblemain import Brain, Wanderer
            from marble.plugins.wanderer_moe_router import MoERouterPlugin

            brain = Brain(1, size=(1,))
            idx = brain.available_indices()[0]
            brain.add_neuron(idx, tensor=[1.0], type_name="autoneuron")

            wanderer = Wanderer(
                brain,
                type_name="moe_router,entropyaware",
                neuroplasticity_type="base",
                seed=0,
            )

            start = brain.neurons[idx]
            wanderer.walk(max_steps=1, start=start)

            router = next(
                plugin
                for plugin in wanderer._wplugins
                if isinstance(plugin, MoERouterPlugin)
            )
            self.assertFalse(router._enabled)
            self.assertTrue(router._experts)
            for name in router._experts:
                self.assertTrue(router.is_active(name))


if __name__ == "__main__":
    unittest.main(verbosity=2)
