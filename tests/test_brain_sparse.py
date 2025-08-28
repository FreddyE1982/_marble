import unittest


class TestBrainSparse(unittest.TestCase):
    def setUp(self):
        from marble.marblemain import Brain
        self.Brain = Brain

    def test_sparse_bounds_closed(self):
        b = self.Brain(2, mode="sparse", sparse_bounds=((0.0, 10.0), (-5.0, 5.0)))

        inside = (5.0, 0.0)
        outside1 = (-1.0, 0.0)
        outside2 = (11.0, 0.0)
        outside3 = (5.0, 6.0)

        print("sparse inside:", b.is_inside(inside))
        self.assertTrue(b.is_inside(inside))
        print("sparse outside1:", b.is_inside(outside1))
        self.assertFalse(b.is_inside(outside1))
        print("sparse outside2:", b.is_inside(outside2))
        self.assertFalse(b.is_inside(outside2))
        print("sparse outside3:", b.is_inside(outside3))
        self.assertFalse(b.is_inside(outside3))

        n1 = b.add_neuron(inside, tensor=[1.0])
        self.assertIsNotNone(n1)
        with self.assertRaises(ValueError):
            b.add_neuron(outside1, tensor=[0.0])

    def test_sparse_min_only_growth(self):
        b = self.Brain(3, mode="sparse", sparse_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, None)))
        inside_low = (0.5, 0.5, 0.0)
        inside_high = (0.5, 0.5, 1000.0)
        outside_low = (0.5, -0.1, 0.0)

        self.assertTrue(b.is_inside(inside_low))
        self.assertTrue(b.is_inside(inside_high))
        self.assertFalse(b.is_inside(outside_low))

        b.add_neuron(inside_low, tensor=[1.0])
        b.add_neuron(inside_high, tensor=[2.0])
        s = b.connect(inside_low, inside_high)
        self.assertIsNotNone(s)

        coords = b.available_indices()
        print("sparse coords count:", len(coords))
        self.assertIn(inside_low, coords)
        self.assertIn(inside_high, coords)


if __name__ == "__main__":
    unittest.main(verbosity=2)
