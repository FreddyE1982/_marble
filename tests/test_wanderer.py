import unittest


class TestWanderer(unittest.TestCase):
    def setUp(self):
        from marble.marblemain import Brain, Wanderer
        self.Brain = Brain
        self.Wanderer = Wanderer

    def test_wanderer_basic_updates(self):
        # Build a tiny sparse brain with two neurons connected bidirectionally
        b = self.Brain(2, mode="sparse", sparse_bounds=((0.0, None), (0.0, None)))
        n1 = b.add_neuron((0.0, 0.0), tensor=[1.0, 2.0], weight=1.0, bias=0.5)
        n2 = b.add_neuron((1.0, 0.0), tensor=[0.5, 1.5], weight=0.8, bias=-0.2)
        b.connect((0.0, 0.0), (1.0, 0.0), direction="bi")

        # Snapshot original weights/biases
        orig = {(id(n1)): (n1.weight, n1.bias), (id(n2)): (n2.weight, n2.bias)}

        w = self.Wanderer(b, seed=123)
        stats = w.walk(max_steps=5, lr=1e-2)
        print("wanderer basic stats:", stats)

        self.assertGreaterEqual(stats["steps"], 1)
        self.assertGreaterEqual(stats["visited"], 1)

        # Ensure at least one visited neuron's params changed
        changed = False
        for vn in w._visited:
            ow, ob = orig[id(vn)] if id(vn) in orig else (None, None)
            if ow is None:
                continue
            if abs(float(vn.weight) - float(ow)) > 1e-9 or abs(float(vn.bias) - float(ob)) > 1e-9:
                changed = True
                break
        self.assertTrue(changed)

    def test_wanderer_plugin_choose_next_called(self):
        from marble.marblemain import register_wanderer_type, Wanderer, Brain

        class DeterministicPlugin:
            def on_init(self, wanderer):
                wanderer._plugin_state["choose_calls"] = 0

            def choose_next(self, wanderer, current, choices):
                # Count calls and choose the first available option deterministically
                wanderer._plugin_state["choose_calls"] += 1
                return choices[0]

        register_wanderer_type("det", DeterministicPlugin())

        b = Brain(2, mode="sparse", sparse_bounds=((0.0, None), (0.0, None)))
        b.add_neuron((0.0, 0.0), tensor=[1.0], weight=1.0, bias=0.0)
        b.add_neuron((1.0, 0.0), tensor=[1.0], weight=1.0, bias=0.0)
        b.connect((0.0, 0.0), (1.0, 0.0), direction="bi")

        w = Wanderer(b, type_name="det", seed=1)
        stats = w.walk(max_steps=4, lr=1e-2)
        print("wanderer plugin choose calls:", w._plugin_state.get("choose_calls", 0))

        # choose_next should be called once per step taken when choices exist (>=1)
        self.assertGreaterEqual(w._plugin_state.get("choose_calls", 0), 1)
        self.assertGreaterEqual(stats["steps"], 1)

    def test_wanderer_exposes_learnable_params(self):
        from marble.marblemain import register_wanderer_type, Wanderer, Brain
        import torch

        class ParamPlugin:
            def loss(self, wanderer, outputs):
                return torch.stack(list(wanderer.learnable_params.values())).sum()

        register_wanderer_type("paramplug", ParamPlugin())
        b = Brain(1, mode="sparse", sparse_bounds=((0.0, None),))
        b.add_neuron((0.0,), tensor=[1.0], weight=1.0, bias=0.0)
        w = Wanderer(b, type_name="paramplug", seed=1)
        self.assertEqual(len(w.learnable_params), 10)
        before = [p.detach().to("cpu").item() for p in w.learnable_params.values()]
        w.walk(max_steps=1, lr=0.1)
        after = [p.detach().to("cpu").item() for p in w.learnable_params.values()]
        self.assertTrue(any(a != b for a, b in zip(after, before)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
