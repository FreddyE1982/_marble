import unittest


class TestPluginStacking(unittest.TestCase):
    def test_brain_train_plugin_stacking(self):
        from marble.marblemain import Brain, Wanderer, register_brain_train_type

        class P1:
            def before_walk(self, brain, wanderer, i):
                return {"max_steps": 1}

        class P2:
            def before_walk(self, brain, wanderer, i):
                return {"lr": 0.001}

        register_brain_train_type("p1", P1())
        register_brain_train_type("p2", P2())

        b = Brain(2, size=(6, 6))
        # Create a simple start neuron so a walk can happen
        it = iter(b.available_indices())
        idx1 = next(it)
        start = b.add_neuron(idx1, tensor=0.0)
        idx2 = next(it)
        dst = b.add_neuron(idx2, tensor=1.0)
        b.connect(idx1, idx2, direction="uni")
        w = Wanderer(b)
        res = b.train(w, num_walks=1, max_steps=5, lr=0.01, type_name="p1,p2", start_selector=lambda brain: start)
        self.assertEqual(len(res["history"]), 1)
        self.assertEqual(res["history"][0]["steps"], 1)  # p1 applied
        # current_lr recorded on wanderer after walk (from p2)
        self.assertAlmostEqual(float(getattr(w, "current_lr", 0.0)), 0.001, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
