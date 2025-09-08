import importlib
import unittest

from marble import plugin_cost_profiler as cp
import marble.wanderer as w
import marble.plugins as p


class PluginTimingWrapperTests(unittest.TestCase):
    def setUp(self) -> None:
        importlib.reload(cp)
        importlib.reload(w)
        importlib.reload(p)
        cp.enable()

    def test_method_records_cost(self) -> None:
        plugin = w.WANDERER_TYPES_REGISTRY["wanderalongsynapseweights"]

        class DummySynapse:
            def __init__(self, weight: float) -> None:
                self.weight = weight

        plugin.choose_next(None, None, [(DummySynapse(1.0), "f"), (DummySynapse(2.0), "b")])
        self.assertGreater(cp.get_cost("wanderalongsynapseweights"), 0.0)


if __name__ == "__main__":
    unittest.main()
