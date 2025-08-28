import unittest

from marble.marblemain import REPORTER, report, report_group, clear_report_group


class TestReporterClear(unittest.TestCase):
    def setUp(self):
        self.reporter = REPORTER
        self.report = report
        self.report_group = report_group
        self.clear = clear_report_group

    def test_clear_group(self):
        r = self.reporter
        r.registergroup("temp", "sub")
        self.report("temp", "val", 42, "sub")
        before = self.report_group("temp", "sub")
        self.assertIn("val", before)
        self.clear("temp", "sub")
        after = self.report_group("temp", "sub")
        print("reporter after clear:", after)
        self.assertEqual(after, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
