import unittest


class TestTripleContrastPlugin(unittest.TestCase):
    def test_triple_contrast_runs(self):
        from marble.marblemain import Brain, Wanderer, report, clear_report_group
        clear_report_group("tests/triple")
        b = Brain(2, size=(3, 3))
        w = Wanderer(b, type_name="triple_contrast", mixedprecision=False)
        stats = w.walk(max_steps=2, lr=0.01)
        report("tests", "triple_contrast_loss", {"loss": stats.get("loss", 0.0)}, "triple")
        print("triple contrast loss:", stats.get("loss", 0.0))
        self.assertIn("loss", stats)


if __name__ == "__main__":
    unittest.main(verbosity=2)
