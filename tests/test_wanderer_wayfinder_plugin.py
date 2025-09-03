import unittest


class TestWayfinderPlugin(unittest.TestCase):
    def setUp(self):
        from marble.marblemain import Brain, Wanderer
        from marble.wanderer import WANDERER_TYPES_REGISTRY
        # ensure plugin module is imported so registration occurs
        from marble.plugins import wanderer_wayfinder  # noqa: F401

        self.Brain = Brain
        self.Wanderer = Wanderer
        self.registry = WANDERER_TYPES_REGISTRY

    def test_map_and_learnables(self):
        b = self.Brain(2, mode="sparse", sparse_bounds=((0.0, None), (0.0, None)))
        n0 = b.add_neuron((0.0, 0.0), tensor=[1.0], weight=1.0, bias=0.0)
        n1 = b.add_neuron((1.0, 0.0), tensor=[1.0], weight=1.0, bias=0.0, connect_to=(0.0, 0.0))
        for s in list(getattr(n1, "outgoing", [])):
            if s.target is n0:
                b.remove_synapse(s)
        n2 = b.add_neuron((0.0, 1.0), tensor=[1.0], weight=1.0, bias=0.0, connect_to=(0.0, 0.0))
        for s in list(getattr(n2, "outgoing", [])):
            if s.target is n0:
                b.remove_synapse(s)
        b.connect((0.0, 0.0), (1.0, 0.0), direction="uni")
        b.connect((0.0, 0.0), (0.0, 1.0), direction="uni")

        w = self.Wanderer(b, type_name="wayfinder", seed=0, loss="nn.MSELoss")
        w.walk(start=n0, max_steps=2, lr=1e-2)

        plugin = self.registry["wayfinder"]
        mp = plugin._maps.get(id(w))
        print("wayfinder map size:", len(mp))
        self.assertTrue(mp)
        self.assertIn("cost_weight", w._learnables)

        # Ensure prune removes nodes when ratio is 1.0 and max_nodes=0
        plugin._prune(mp, 1.0, 0)
        self.assertEqual(len(mp), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

