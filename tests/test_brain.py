import unittest


class TestBrain(unittest.TestCase):
    def setUp(self):
        from marble.marblemain import Brain, Neuron

        self.Brain = Brain
        self.Neuron = Neuron

    def test_circle_formula_2d(self):
        # Circle: n1^2 + n2^2 <= 1 within bounds [-1,1]
        b = self.Brain(2, size=(11, 11), bounds=((-1.0, 1.0), (-1.0, 1.0)), formula="n1*n1 + n2*n2 <= 1.0")

        center = (5, 5)
        print("brain 2d circle center inside:", b.is_inside(center))
        self.assertTrue(b.is_inside(center))
        corner = (0, 0)
        print("brain 2d circle corner outside:", b.is_inside(corner))
        self.assertFalse(b.is_inside(corner))

        # Place a neuron inside and try placing outside
        n_in = b.add_neuron(center, tensor=[1.0, 2.0], weight=2.0, bias=0.0)
        self.assertIsNotNone(n_in)
        with self.assertRaises(ValueError):
            b.add_neuron(corner, tensor=[0.0])

        # Connect two neurons inside
        p2 = (6, 5)
        b.add_neuron(p2, tensor=[0.0])
        s = b.connect(center, p2)
        print("brain add synapse count:", len(b.synapses))
        self.assertIn(s, b.synapses)

    def test_mandelbrot_default_2d(self):
        # With no formula and n=2, default uses Mandelbrot occupancy
        b = self.Brain(2, size=(8, 8))
        all_idx = b.available_indices()
        print("brain mandelbrot available indices:", len(all_idx))
        self.assertGreater(len(all_idx), 0)
        # Place a neuron at some available index
        idx = all_idx[len(all_idx) // 2]
        n = b.add_neuron(idx, tensor=[0.0])
        self.assertIsNotNone(n)

    def test_nd_formula_3d(self):
        # 3D sphere: n1^2 + n2^2 + n3^2 <= 1
        b = self.Brain(3, size=(7, 7, 7), bounds=((-1, 1), (-1, 1), (-1, 1)), formula="n1*n1 + n2*n2 + n3*n3 <= 1.0")
        center = (3, 3, 3)
        print("brain 3d sphere center inside:", b.is_inside(center))
        self.assertTrue(b.is_inside(center))
        outside = (0, 0, 0)
        print("brain 3d sphere corner outside:", b.is_inside(outside))
        self.assertFalse(b.is_inside(outside))
        b.add_neuron(center, tensor=[1.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
