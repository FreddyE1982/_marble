import unittest


class TestReporter(unittest.TestCase):
    def setUp(self):
        from marble.marblemain import Reporter, REPORTER
        self.Reporter = Reporter
        self.reporter = REPORTER

    def test_register_and_set_item(self):
        r = self.reporter
        r.registergroup("metrics")
        r.item["loss", "metrics"] = 1.23
        r.item["acc", "metrics"] = [0.9, 0.92]
        # update existing
        r.item["loss", "metrics"] = 1.11
        g = r.group("metrics")
        print("reporter metrics group:", g)
        self.assertIn("loss", g)
        self.assertEqual(g["loss"], 1.11)
        self.assertIn("acc", g)
        self.assertEqual(g["acc"], [0.9, 0.92])

    def test_get_item_call(self):
        r = self.reporter
        r.registergroup("debug")
        r.item["msg", "debug"] = {"a": 1}
        val = r.item("msg", "debug")
        print("reporter debug msg:", val)
        self.assertEqual(val, {"a": 1})


if __name__ == "__main__":
    unittest.main(verbosity=2)

