import unittest

import marble.plugins  # ensure plugin discovery
from marble.marblemain import Brain, Wanderer, register_wanderer_type
from marble.plugins.wanderer_autoplugin import AutoPlugin


class TestAutoPluginDecisionInterval(unittest.TestCase):
    def test_interval_respected(self):
        steps = []
        original = AutoPlugin.is_active

        def spy(self, wanderer, name, neuron, plugintype="wanderer"):
            step = getattr(wanderer, "neuron_fire_count", 0)
            res = original(self, wanderer, name, neuron, plugintype)
            if name == "EpsilonGreedyChooserPlugin" and step % self._decision_interval == 0:
                steps.append(step)
            return res

        AutoPlugin.is_active = spy  # type: ignore[assignment]
        register_wanderer_type(
            "auto_interval",
            AutoPlugin(decision_interval=2, disabled_plugins=["ResourceAllocatorPlugin"]),
        )
        b = Brain(2, mode="sparse", sparse_bounds=((0.0, None), (0.0, None)))
        n1 = b.add_neuron((0.0, 0.0), tensor=[1.0], weight=1.0, bias=0.0)
        n2 = b.add_neuron((1.0, 0.0), tensor=[1.0], weight=1.0, bias=0.0, connect_to=(0.0, 0.0))
        n3 = b.add_neuron((0.0, 1.0), tensor=[1.0], weight=1.0, bias=0.0, connect_to=(0.0, 0.0))
        b.connect((0.0, 0.0), (1.0, 0.0), direction="bi")
        b.connect((0.0, 0.0), (0.0, 1.0), direction="bi")
        w = Wanderer(b, type_name="epsilongreedy,auto_interval", neuroplasticity_type="base", seed=0)
        w.walk(max_steps=5, start=n1, lr=0.01)
        print("recompute steps:", steps)
        self.assertTrue(all(s % 2 == 0 for s in steps))
        self.assertTrue(all(b - a == 2 for a, b in zip(steps, steps[1:])))


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
