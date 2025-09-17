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

    def test_dict_merge(self):
        r = self.reporter
        r.registergroup("stats")
        r.item["status", "stats"] = {"a": 1}
        # second set should merge rather than replace
        r.item["status", "stats"] = {"b": 2}
        merged = r.group("stats")["status"]
        print("reporter merged status:", merged)
        self.assertEqual(merged, {"a": 1, "b": 2})

    def test_tensorboard_logging(self):
        r = self.reporter
        logdir = r.tensorboard_logdir()
        self.assertIsNotNone(logdir)
        logdir = str(logdir)
        import pathlib

        path = pathlib.Path(logdir)
        path.mkdir(parents=True, exist_ok=True)
        before = {p: p.stat().st_size for p in path.glob("events.*")}
        r.item["tb_scalar", "tensorboard"] = 3.14
        r.item["tb_hist", "tensorboard"] = [1.0, 2.0, 3.5]
        r.item["tb_text", "tensorboard"] = {"nested": "value"}
        r.flush_tensorboard()
        after = {p: p.stat().st_size for p in path.glob("events.*")}
        print("tensorboard files before:", before)
        print("tensorboard files after:", after)
        self.assertTrue(after, "TensorBoard writer did not create any event files")
        new_files = set(after) - set(before)
        if new_files:
            self.assertTrue(all(after[p] > 0 for p in new_files))
        else:
            self.assertTrue(any(after[p] > before[p] for p in after))


if __name__ == "__main__":
    unittest.main(verbosity=2)

