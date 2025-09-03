import unittest


class TestWandererPreForward(unittest.TestCase):
    def test_pre_forward_option_measures_transfer(self):
        from marble.marblemain import Brain, Wanderer
        b = Brain(2, size=(4, 4))
        it = iter(b.available_indices())
        i1 = next(it)
        i2 = next(it)
        n1 = b.add_neuron(i1, tensor=0.0)
        n2 = b.add_neuron(i2, tensor=0.0, connect_to=i1, direction="uni")
        # remove auto connection from n2->n1 and reconnect i1->i2
        for s in list(getattr(n2, "outgoing", [])):
            if s.target is n1:
                b.remove_synapse(s)
        b.connect(i1, i2, direction="uni")
        w = Wanderer(b, pre_forward=True)
        w.walk(max_steps=1, start=n1, lr=1e-2)
        print("pre_forward times:", w._last_pre_forward_time, w._last_transfer_time)
        self.assertTrue(w.pre_forward)
        self.assertGreaterEqual(w._last_transfer_time, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
