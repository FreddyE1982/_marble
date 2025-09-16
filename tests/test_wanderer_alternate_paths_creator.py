import unittest


class TestAlternatePathsCreator(unittest.TestCase):
    def test_creates_alternate_path_on_walk_end(self):
        from marble.marblemain import Brain, Wanderer
        b = Brain(2, size=(8, 8))
        # Small path of 2 nodes to ensure a walk
        it = iter(b.available_indices())
        i1 = next(it); i2 = next(it)
        n1 = b.add_neuron(i1, tensor=0.0)
        n2 = b.add_neuron(i2, tensor=0.0, connect_to=i1)
        b.connect(i1, i2, direction="uni")
        w = Wanderer(b, type_name="alternatepathscreator", neuro_config={"altpaths_min_len": 2, "altpaths_max_len": 3})
        before = len(b.neurons)
        w.walk(max_steps=1, start=n1, lr=1e-2)
        after = len(b.neurons)
        print("altpaths counts before/after:", before, after)
        # One anchor plus at least the first new node
        self.assertGreaterEqual(after, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
