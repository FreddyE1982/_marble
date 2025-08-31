import unittest


class TestResourceAllocatorPlugin(unittest.TestCase):
    def setUp(self):
        from marble.marblemain import Brain, Wanderer
        self.Brain = Brain
        self.Wanderer = Wanderer

    def test_plugin_active_by_default_and_params(self):
        b = self.Brain(1, size=(4,))
        n1 = b.add_neuron((0,), tensor=0.0)
        n2 = b.add_neuron((1,), tensor=0.0)
        b.connect(getattr(n1, "position"), getattr(n2, "position"), direction="uni")
        w = self.Wanderer(b)
        names = [p.__class__.__name__ for p in getattr(w, "_wplugins", [])]
        print("plugins", names)
        self.assertIn("ResourceAllocatorPlugin", names)
        plug = next(p for p in w._wplugins if p.__class__.__name__ == "ResourceAllocatorPlugin")
        params = plug._params(w)
        print("param_count", len(params))
        self.assertGreaterEqual(len(params), 30)


if __name__ == "__main__":
    unittest.main(verbosity=2)
