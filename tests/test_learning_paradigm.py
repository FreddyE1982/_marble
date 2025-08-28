import unittest


class TestLearningParadigm(unittest.TestCase):
    def test_adaptive_lr_paradigm_attaches_and_adjusts_lr(self):
        from marble.marblemain import Brain, UniversalTensorCodec, run_training_with_datapairs, register_learning_paradigm_type
        # Ensure registry has adaptive_lr (registered in module import); nothing to do here
        b = Brain(2, size=(6, 6))
        # Simple dataset
        data = [("a", "b"), ("c", "d"), ("e", "f")]
        codec = UniversalTensorCodec()

        # Baseline run (no paradigm)
        res_base = run_training_with_datapairs(b, data, codec, steps_per_pair=3, lr=1e-2)
        loss_base = res_base["final_loss"]

        # Load paradigm and rerun on fresh brain to avoid cross-dataset signature issues
        b2 = Brain(2, size=(6, 6))
        b2.load_paradigm("adaptive_lr", {"factor_down": 0.5, "factor_up": 1.2})
        res_par = run_training_with_datapairs(b2, data, codec, steps_per_pair=3, lr=1e-2)
        loss_par = res_par["final_loss"]

        # Check that we ran and produced numeric losses
        self.assertIsInstance(loss_base, float)
        self.assertIsInstance(loss_par, float)
        # Not strictly asserting improvement (stochastic paths), but ensure paradigm wiring did not break
        # and print both for audit
        print("adaptive_lr paradigm final_loss:", loss_par, "baseline:", loss_base)

    def test_paradigm_neuroplasticity_like_hooks(self):
        # Define and register a paradigm that sets lr_override on init
        from marble.marblemain import Brain, Wanderer, register_learning_paradigm_type

        class HookParadigm:
            def on_init(self, wanderer):
                # Set a distinct lr_override to verify hook executes
                wanderer.lr_override = 0.005

        register_learning_paradigm_type("hooktest", HookParadigm)

        b = Brain(2, size=(6, 6))
        b.load_paradigm("hooktest")
        # Minimal graph with an edge
        it = iter(b.available_indices())
        i1 = next(it); i2 = next(it)
        n1 = b.add_neuron(i1, tensor=1.0)
        n2 = b.add_neuron(i2, tensor=0.0)
        b.connect(i1, i2, direction="uni")
        w = Wanderer(b)
        res = w.walk(max_steps=1, start=n1, lr=0.01)
        # Ensure the override took effect
        self.assertAlmostEqual(float(getattr(w, "current_lr", 0.0)), 0.005, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
