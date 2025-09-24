import unittest

from marble.plugin_telemetry import (
    get_plugin_catalog,
    get_plugin_usage,
    record_plugin_activation,
    reset_plugin_usage,
)
from marble.reporter import REPORTER


class PluginTelemetryTest(unittest.TestCase):
    def setUp(self) -> None:
        reset_plugin_usage()

    def test_catalog_contains_core_plugins(self) -> None:
        catalogue = get_plugin_catalog()
        self.assertIn("conv1d", catalogue)
        conv1d = catalogue["conv1d"]
        self.assertEqual(conv1d["plugin_type"], "neuron")
        self.assertTrue(conv1d["niche"])
        self.assertIsInstance(conv1d["hooks"], list)
        self.assertGreater(len(conv1d["hooks"]), 0)

    def test_record_updates_usage_and_reporter(self) -> None:
        record_plugin_activation("unit_test_plugin", "forward", 0.005)
        usage = get_plugin_usage()
        self.assertIn("unit_test_plugin", usage)
        stats = usage["unit_test_plugin"]
        self.assertEqual(stats["calls"], 1)
        self.assertGreater(stats["avg_latency_ms"], 0.0)
        group_payload = REPORTER.group("plugins", "metrics")
        self.assertIn("usage", group_payload)
        reporter_usage = group_payload["usage"]
        self.assertIn("unit_test_plugin", reporter_usage)
        self.assertEqual(reporter_usage["unit_test_plugin"]["calls"], 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
