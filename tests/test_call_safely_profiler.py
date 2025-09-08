import importlib
import time
import unittest

import marble.marblemain as mm
from marble import plugin_cost_profiler as cp


class CallSafelyProfilerTests(unittest.TestCase):
    def setUp(self) -> None:
        importlib.reload(cp)
        importlib.reload(mm)

    def test_records_cost_when_plugin_name_provided(self) -> None:
        def dummy() -> None:
            time.sleep(0.01)

        mm._call_safely(dummy, plugin_name="dummy")
        cost = cp.get_cost("dummy")
        print("recorded cost", cost)
        self.assertGreater(cost, 0.0)


if __name__ == "__main__":
    unittest.main()
