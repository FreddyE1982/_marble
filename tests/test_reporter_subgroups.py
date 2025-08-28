import unittest


class TestReporterSubgroups(unittest.TestCase):
    def setUp(self):
        from marble.marblemain import REPORTER, report, report_group, report_dir
        self.REPORTER = REPORTER
        self.report = report
        self.report_group = report_group
        self.report_dir = report_dir

    def test_subgroup_set_and_get(self):
        r = self.REPORTER
        r.registergroup("metrics", "epoch1")
        self.report("metrics", "loss", 0.9, "epoch1")
        self.report("metrics", "acc", [0.8, 0.85], "epoch1")
        items = self.report_group("metrics", "epoch1")
        print("reporter subgroup metrics/epoch1:", items)
        self.assertEqual(items["loss"], 0.9)
        self.assertEqual(items["acc"], [0.8, 0.85])

    def test_dir_and_tree(self):
        r = self.REPORTER
        r.registergroup("logs", "train", "epoch2")
        self.report("logs", "msg", {"ok": True}, "train", "epoch2")
        tops = r.dirgroups()
        print("reporter top groups:", tops)
        self.assertIn("logs", tops)
        tree = self.report_dir("logs")
        print("reporter tree logs:", tree)
        self.assertIn("train", tree["subgroups"]) 


if __name__ == "__main__":
    unittest.main(verbosity=2)

