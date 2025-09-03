import unittest

from marble.graph import Neuron
from marble.reporter import REPORTER, clear_report_group


class TestDeviceReporting(unittest.TestCase):
    def setUp(self):
        clear_report_group("neuron")

    def test_forward_reports_device(self):
        n = Neuron([1.0])
        n._device = "cuda"
        n.forward(n.tensor)
        item = REPORTER.item("forward", "neuron", "metrics")
        print("device in neuron forward report:", item.get("device"))
        self.assertEqual(item.get("device"), "cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
