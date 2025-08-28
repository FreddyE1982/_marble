import unittest


class TestLearningParadigmToggleAndGrowth(unittest.TestCase):
    def test_enable_disable_paradigm(self):
        from marble.marblemain import Brain, Wanderer, add_paradigm, apply_paradigms_to_wanderer
        b = Brain(2, size=(6, 6))
        p = add_paradigm(b, "adaptive_lr", {"factor_down": 0.5})
        # Disable it before applying
        b.enable_paradigm(p, enabled=False)
        w = Wanderer(b)
        apply_paradigms_to_wanderer(b, w)
        # Should not have any self-attention attached from the paradigm
        self.assertEqual(len(getattr(w, "_selfattentions", []) or []), 0)
        # Enable and apply again on a new wanderer
        b.enable_paradigm(p, enabled=True)
        w2 = Wanderer(b)
        apply_paradigms_to_wanderer(b, w2)
        self.assertGreaterEqual(len(getattr(w2, "_selfattentions", []) or []), 1)

    def test_growth_paradigm_grows_when_stuck(self):
        from marble.marblemain import Brain, Wanderer
        b = Brain(2, size=(6, 6))
        # Load growth paradigm
        b.load_paradigm("growth", {"grow_on_step_when_stuck": True, "grow_on_end_if_no_outgoing": True, "max_new_per_walk": 1})
        # Single neuron, no outgoing
        idx = next(iter(b.available_indices()))
        n0 = b.add_neuron(idx, tensor=0.0)
        w = Wanderer(b)
        # Walk 1 step from the single neuron; paradigm should attach a new neuron
        before = len(b.neurons)
        w.walk(max_steps=1, start=n0, lr=1e-2)
        after = len(b.neurons)
        print("growth neurons before/after:", before, after)
        self.assertGreaterEqual(after, before + 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

