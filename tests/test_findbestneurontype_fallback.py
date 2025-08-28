import unittest

class TestFindBestNeuronTypeFallback(unittest.TestCase):
    def setUp(self):
        from marble.marblemain import Brain, Wanderer, SelfAttention, attach_selfattention
        from marble.plugins.selfattention_findbestneurontype import FindBestNeuronTypeRoutine
        from marble.reporter import report
        self.Brain = Brain
        self.Wanderer = Wanderer
        self.SelfAttention = SelfAttention
        self.attach = attach_selfattention
        self.Routine = FindBestNeuronTypeRoutine
        self.report = report

    def test_fallback_to_base(self):
        b = self.Brain(1, size=(3,))
        w = self.Wanderer(b, seed=1)
        sa = self.SelfAttention(routines=[self.Routine()])
        self.attach(w, sa)
        idx = b.available_indices()[0]
        n = b.add_neuron(idx, tensor=[0.0])
        self.report("test", "findbestneurontype_fallback", {"type": n.type_name}, "events")
        print("fallback neuron type:", n.type_name)
        self.assertIsNone(n.type_name)

if __name__ == "__main__":
    unittest.main(verbosity=2)
