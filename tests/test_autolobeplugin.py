import unittest


class TestAutoLobePlugin(unittest.TestCase):
    def test_creates_lobes_and_learnable_param(self):
        from marble.marblemain import Brain, Wanderer, clear_report_group, report

        clear_report_group("tests/autolobe")

        b = Brain(1, size=(2,))
        idx0, idx1 = b.available_indices()
        n0 = b.add_neuron(idx0, tensor=[0.0], type_name="autoneuron")
        b.add_neuron(idx1, tensor=[0.0], type_name="autoneuron", connect_to=idx0, direction="uni")

        w = Wanderer(b, type_name="autolobe", neuroplasticity_type="base", seed=0)
        auto = next(p for p in w._wplugins if p.__class__.__name__ == "AutoLobePlugin")
        auto.before_walk(w, n0)

        self.assertIn("autolobe_low", b.lobes)
        self.assertIn("autolobe_high", b.lobes)
        self.assertIn("autolobe_threshold", w._learnables)
        report("tests/autolobe", "done", {"lobes": list(b.lobes.keys())}, "tests")
        print("lobes created:", list(b.lobes.keys()))


if __name__ == "__main__":
    unittest.main(verbosity=2)

