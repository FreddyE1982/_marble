import unittest


class TestWandererWalkSummary(unittest.TestCase):
    def setUp(self):
        from marble.marblemain import (
            Brain,
            Wanderer,
            get_last_walk_summary,
            clear_report_group,
            report,
        )
        self.Brain = Brain
        self.Wanderer = Wanderer
        self.get_last_walk_summary = get_last_walk_summary
        self.clear_report_group = clear_report_group
        self.report = report

    def test_walk_summary_recorded(self):
        # Ensure clean reporter state
        self.clear_report_group("training")

        b = self.Brain(2, mode="sparse", sparse_bounds=((0.0, None), (0.0, None)))
        b.add_neuron((0.0, 0.0), tensor=[1.0], weight=1.0, bias=0.0)
        b.add_neuron((1.0, 0.0), tensor=[1.0], weight=1.0, bias=0.0)
        b.connect((0.0, 0.0), (1.0, 0.0), direction="bi")

        w = self.Wanderer(b, seed=123)
        w.walk(max_steps=2, lr=1e-2)

        summary = self.get_last_walk_summary()
        print("latest walk summary:", summary)
        self.report("tests", "walk_summary", summary)
        self.assertIsNotNone(summary)
        self.assertIn("final_loss", summary)
        self.assertIn("steps", summary)
        self.assertGreaterEqual(summary["steps"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
