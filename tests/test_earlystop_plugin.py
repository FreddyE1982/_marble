import types
import unittest


class TestEarlyStopPlugin(unittest.TestCase):
    def setUp(self):
        from marble.marblemain import clear_report_group

        self.clear = clear_report_group
        self.clear("training")

    def test_stops_training_early(self):
        from marble.marblemain import Brain, Wanderer, report

        brain = Brain(1, size=2)
        w = Wanderer(brain)

        def constant_walk(self, max_steps=1, start=None, lr=1e-2):
            return {"loss": 1.0, "steps": max_steps}

        w.walk = types.MethodType(constant_walk, w)
        res = brain.train(w, num_walks=10, max_steps=1, lr=0.1, type_name="earlystop")
        report("tests", "earlystop", {"history_len": len(res["history"])}, "plugin")
        print("history len and early stopped:", len(res["history"]), res.get("early_stopped"))
        self.assertLessEqual(len(res["history"]), 4)
        self.assertTrue(res.get("early_stopped"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

