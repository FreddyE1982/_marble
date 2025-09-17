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

    def test_tensorboard_training_datapair_aggregates(self):
        from marble.reporter import _TensorBoardAdapter

        class DummyWriter:
            def __init__(self):
                self.scalars = []
                self.histograms = []
                self.text = []
                self.log_dir = "dummy"

            def add_scalar(self, tag, value, step):
                self.scalars.append((tag, value, step))

            def add_histogram(self, tag, tensor, step):
                self.histograms.append((tag, tensor, step))

            def add_text(self, tag, text, step):
                self.text.append((tag, text, step))

            def flush(self):
                pass

        adapter = _TensorBoardAdapter(True, None, 10)
        adapter._writer = DummyWriter()

        adapter.log(("training", "datapair"), "pair_1", {"loss": 0.75, "steps": 12})
        adapter.log(("training", "datapair"), "pair_2", {"loss": 0.5, "steps": 8})

        loss_entries = [entry for entry in adapter._writer.scalars if entry[0] == "training/datapair/loss"]
        step_entries = [entry for entry in adapter._writer.scalars if entry[0] == "training/datapair/steps"]

        self.assertEqual([0.75, 0.5], [entry[1] for entry in loss_entries])
        self.assertEqual([0, 1], [entry[2] for entry in loss_entries])
        self.assertEqual([12.0, 8.0], [entry[1] for entry in step_entries])
        self.assertEqual([0, 1], [entry[2] for entry in step_entries])
        self.assertFalse(any("pair_" in tag for tag, _, _ in adapter._writer.scalars))

        before = len(adapter._writer.scalars)
        adapter.log(("wanderer_steps", "logs"), "chunk", {"loss": 1.0})
        self.assertEqual(before, len(adapter._writer.scalars))

    def test_tensorboard_training_walks_aggregates(self):
        from marble.reporter import _TensorBoardAdapter

        class DummyWriter:
            def __init__(self):
                self.scalars = []
                self.histograms = []
                self.text = []
                self.log_dir = "dummy"

            def add_scalar(self, tag, value, step):
                self.scalars.append((tag, value, step))

            def add_histogram(self, tag, tensor, step):
                self.histograms.append((tag, tensor, step))

            def add_text(self, tag, text, step):
                self.text.append((tag, text, step))

            def flush(self):
                pass

        adapter = _TensorBoardAdapter(True, None, 10)
        adapter._writer = DummyWriter()

        adapter.log(
            ("training", "walks"),
            "walk_1",
            {"steps": 10, "final_loss": 0.5, "timestamp": 1.23},
        )
        adapter.log(
            ("training", "walks"),
            "walk_2",
            {"steps": 12, "final_loss": 0.25, "timestamp": 2.34},
        )

        tags = [tag for tag, _, _ in adapter._writer.scalars]
        self.assertTrue(tags, "Expected aggregated walk scalars to be logged")
        self.assertTrue(all("walk_" not in tag for tag in tags))

        steps_entries = [entry for entry in adapter._writer.scalars if entry[0] == "training/walks/steps"]
        loss_entries = [entry for entry in adapter._writer.scalars if entry[0] == "training/walks/final_loss"]
        timestamp_entries = [entry for entry in adapter._writer.scalars if entry[0] == "training/walks/timestamp"]

        self.assertEqual([10.0, 12.0], [entry[1] for entry in steps_entries])
        self.assertEqual([0, 1], [entry[2] for entry in steps_entries])
        self.assertEqual([0.5, 0.25], [entry[1] for entry in loss_entries])
        self.assertEqual([0, 1], [entry[2] for entry in loss_entries])
        self.assertEqual([1.23, 2.34], [entry[1] for entry in timestamp_entries])
        self.assertEqual([0, 1], [entry[2] for entry in timestamp_entries])


if __name__ == "__main__":
    unittest.main(verbosity=2)

