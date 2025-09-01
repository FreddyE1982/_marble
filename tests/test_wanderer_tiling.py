import unittest


class TestWandererTiling(unittest.TestCase):
    def setUp(self):
        from marble.marblemain import Brain, Wanderer
        self.Brain = Brain
        self.Wanderer = Wanderer

    def test_wanderer_tiling_basic(self):
        b = self.Brain(2, mode="sparse", sparse_bounds=((0.0, None), (0.0, None)))
        n1 = b.add_neuron((0.0, 0.0), tensor=[1.0, 2.0], weight=1.0, bias=0.5)
        n2 = b.add_neuron((1.0, 0.0), tensor=[0.5, 1.5], weight=0.8, bias=-0.2)
        b.connect((0.0, 0.0), (1.0, 0.0), direction="bi")

        orig = {id(n1): (n1.weight, n1.bias), id(n2): (n2.weight, n2.bias)}

        w = self.Wanderer(b, seed=123, tiling=True, tile_size=2)
        stats = w.walk(max_steps=5, lr=1e-2)
        print("wanderer tiling stats:", stats)

        self.assertGreaterEqual(stats["steps"], 1)
        self.assertGreaterEqual(stats["visited"], 1)

        changed = False
        for vn in w._visited:
            ow, ob = orig.get(id(vn), (None, None))
            if ow is None:
                continue
            if abs(float(vn.weight) - float(ow)) > 1e-9 or abs(float(vn.bias) - float(ob)) > 1e-9:
                changed = True
                break
        self.assertTrue(changed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
