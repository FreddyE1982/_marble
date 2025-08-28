import unittest


class TestWandererFreeze(unittest.TestCase):
    def setUp(self):
        from marble.marblemain import Brain, Wanderer, register_wanderer_type, report

        self.Brain = Brain
        self.Wanderer = Wanderer
        self.register = register_wanderer_type
        self.report = report

    def test_freeze_single_neuron(self):
        b = self.Brain(2, mode="sparse", sparse_bounds=((0.0, None), (0.0, None)))
        n1 = b.add_neuron((0.0, 0.0), tensor=[1.0], weight=1.0, bias=0.0)
        n2 = b.add_neuron((1.0, 0.0), tensor=[1.0], weight=1.0, bias=0.0)
        b.connect((0.0, 0.0), (1.0, 0.0), direction="bi")

        class FreezePlugin:
            def before_walk(self, wanderer, start):
                wanderer.freeze_neuron(n2)

        self.register("freeze_neuron", FreezePlugin())

        w = self.Wanderer(b, type_name="freeze_neuron", seed=1)
        orig = (n2.weight, n2.bias)
        self.report("test", "freeze_single_neuron_start", {"w": orig[0], "b": orig[1]}, "events")
        w.walk(max_steps=3, lr=1e-2)
        self.report("test", "freeze_single_neuron_end", {"w": n2.weight, "b": n2.bias}, "events")
        self.assertEqual((n2.weight, n2.bias), orig)

    def test_freeze_path(self):
        b = self.Brain(2, mode="sparse", sparse_bounds=((0.0, None), (0.0, None)))
        n1 = b.add_neuron((0.0, 0.0), tensor=[1.0], weight=1.0, bias=0.0)
        n2 = b.add_neuron((1.0, 0.0), tensor=[1.0], weight=1.0, bias=0.0)
        s = b.connect((0.0, 0.0), (1.0, 0.0), direction="bi")

        class PathFreezePlugin:
            def before_walk(self, wanderer, start):
                wanderer.freeze_path([s])

        self.register("freeze_path", PathFreezePlugin())

        w = self.Wanderer(b, type_name="freeze_path", seed=2)
        stats = w.walk(max_steps=3, lr=1e-2)
        self.report("test", "freeze_path_stats", stats, "events")
        self.assertEqual(stats["steps"], 0)
        self.assertEqual(len(w._visited), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

