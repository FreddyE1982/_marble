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

    def test_method_records_cost(self) -> None:
        plugin = w.WANDERER_TYPES_REGISTRY["wanderalongsynapseweights"]

        class DummySynapse:
            def __init__(self, weight: float) -> None:
                self.weight = weight

        plugin.choose_next(None, None, [(DummySynapse(1.0), "f"), (DummySynapse(2.0), "b")])
        self.assertGreater(cp.get_cost("wanderalongsynapseweights"), 0.0)

    def test_return_value_preserved(self) -> None:
        class Dummy:
            def echo(self, x: int) -> int:
                return x

        inst = Dummy()
        p._wrap_public_methods(inst, "dummy")
        self.assertEqual(inst.echo(123), 123)
        self.assertGreater(cp.get_cost("dummy"), 0.0)

    def test_exception_passthrough(self) -> None:
        class Dummy:
            def boom(self) -> None:
                raise RuntimeError("boom")

        inst = Dummy()
        p._wrap_public_methods(inst, "dummy")
        with self.assertRaises(RuntimeError):
            inst.boom()
        self.assertGreater(cp.get_cost("dummy"), 0.0)


if __name__ == "__main__":
    unittest.main()
