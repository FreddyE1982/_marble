import unittest


class TestCurriculumAndTempPlugins(unittest.TestCase):
    def test_curriculum_increases_steps(self):
        from marble.marblemain import Brain, Wanderer, register_brain_train_type
        b = Brain(2, size=(6, 6))
        # minimal path
        it = iter(b.available_indices())
        i1 = next(it); i2 = next(it)
        n1 = b.add_neuron(i1, tensor=0.0)
        n2 = b.add_neuron(i2, tensor=0.0, connect_to=i1, direction="uni")
        for s in list(getattr(n2, "outgoing", [])):
            if s.target is n1:
                b.remove_synapse(s)
        b.connect(i1, i2, direction="uni")
        w = Wanderer(b)
        # Use curriculum
        res = b.train(w, num_walks=2, max_steps=1, lr=1e-2, type_name="curriculum", start_selector=lambda brain: n1)
        steps = [h.get("steps", 0) for h in res["history"]]
        print("curriculum steps:", steps)
        self.assertTrue(steps[1] >= steps[0])

    def test_temporary_plugin_stack(self):
        from marble.marblemain import Brain, Wanderer, push_temporary_plugins, pop_temporary_plugins
        b = Brain(2, size=(6, 6))
        w = Wanderer(b)
        prev_len = len(getattr(w, "_wplugins", []) or [])
        handle = push_temporary_plugins(w, wanderer_types=["epsilongreedy", "wanderalongsynapseweights"], neuro_types=["base"])  # base may be redundant
        now_len = len(getattr(w, "_wplugins", []) or [])
        self.assertGreaterEqual(now_len, prev_len + 2)
        pop_temporary_plugins(w, handle)
        after_len = len(getattr(w, "_wplugins", []) or [])
        self.assertEqual(after_len, prev_len)


if __name__ == "__main__":
    unittest.main(verbosity=2)
