import unittest


class TestWandererBestPathAndWeights(unittest.TestCase):
    def setUp(self):
        from marble.marblemain import Brain, Wanderer
        self.Brain = Brain
        self.Wanderer = Wanderer

    def _add_two_paths(self, b):
        it = iter(b.available_indices())
        s = b.add_neuron(next(it), tensor=0.0)
        a = b.add_neuron(next(it), tensor=[0.1])  # low loss path
        bnode = b.add_neuron(next(it), tensor=[1.5])  # higher loss path
        # Downstream nodes to receive
        da = b.add_neuron(next(it), tensor=0.0)
        db = b.add_neuron(next(it), tensor=0.0)
        sa = b.connect(getattr(s, "position"), getattr(a, "position"), direction="uni")
        sb = b.connect(getattr(s, "position"), getattr(bnode, "position"), direction="uni")
        b.connect(getattr(a, "position"), getattr(da, "position"), direction="uni")
        b.connect(getattr(bnode, "position"), getattr(db, "position"), direction="uni")
        return s, sa, sb, a, bnode

    def test_wander_along_weights_prefers_higher(self):
        b = self.Brain(2, size=(8, 8))
        s, sa, sb, a, bnode = self._add_two_paths(b)
        # Set weights to prefer B path
        sa.weight = 0.1
        sb.weight = 2.0
        w = self.Wanderer(b, type_name="wanderalongsynapseweights")
        res = w.walk(max_steps=2, start=s, lr=1e-2)
        # After one step, visited should include the target of sb (bnode)
        self.assertIn(bnode, getattr(w, "_visited", []))

    def test_bestlosspath_updates_weights_then_weights_plugin_prefers(self):
        b = self.Brain(2, size=(8, 8))
        s, sa, sb, a, bnode = self._add_two_paths(b)
        # Initially equal weights
        sa.weight = 1.0
        sb.weight = 1.0
        # Stack plugins: bestlosspath runs first, weights chooser last
        w = self.Wanderer(b, type_name="bestlosspath,wanderalongsynapseweights")
        res = w.walk(max_steps=2, start=s, lr=1e-2)
        # best path is via 'a' (lower loss), so sa weight should be boosted above sb
        self.assertGreater(sa.weight, sb.weight)
        # And the visited should include 'a'
        self.assertIn(a, getattr(w, "_visited", []))


if __name__ == "__main__":
    unittest.main(verbosity=2)
