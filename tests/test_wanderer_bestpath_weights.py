import unittest


class TestWandererBestPathAndWeights(unittest.TestCase):
    def setUp(self):
        from marble.marblemain import Brain, Wanderer
        self.Brain = Brain
        self.Wanderer = Wanderer

    def _add_two_paths(self, b):
        it = iter(b.available_indices())
        s_idx = next(it)
        a_idx = next(it)
        b_idx = next(it)
        da_idx = next(it)
        db_idx = next(it)

        s = b.add_neuron(s_idx, tensor=0.0)
        b.add_neuron(da_idx, tensor=0.0, connect_to=s_idx)
        b.remove_synapse(b.synapses[-1])
        a = b.add_neuron(a_idx, tensor=[0.1], connect_to=da_idx, direction="uni")  # low loss path
        b.add_neuron(db_idx, tensor=0.0, connect_to=s_idx)
        b.remove_synapse(b.synapses[-1])
        bnode = b.add_neuron(b_idx, tensor=[1.5], connect_to=db_idx, direction="uni")  # higher loss path
        sa = b.connect(s_idx, a_idx, direction="uni")
        sb = b.connect(s_idx, b_idx, direction="uni")
        b.connect(a_idx, da_idx, direction="uni")
        b.connect(b_idx, db_idx, direction="uni")
        return s, sa, sb, a, bnode

    def test_wander_along_weights_prefers_higher(self):
        b = self.Brain(2, size=(8, 8))
        s, sa, sb, a, bnode = self._add_two_paths(b)
        # Set weights to prefer B path
        sa.weight = 0.1
        sb.weight = 2.0
        w = self.Wanderer(b, type_name="wanderalongsynapseweights")
        res = w.walk(max_steps=2, start=s, lr=1e-2)
        print("visited after weights:", getattr(w, "_visited", []))
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
        print("weights after best path:", sa.weight, sb.weight)
        # best path is via 'a' (lower loss), so sa weight should be boosted above sb
        self.assertGreater(sa.weight, sb.weight)
        # And the visited should include 'a'
        print("visited after best path:", getattr(w, "_visited", []))
        self.assertIn(a, getattr(w, "_visited", []))


if __name__ == "__main__":
    unittest.main(verbosity=2)
