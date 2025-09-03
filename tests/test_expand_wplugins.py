import unittest

from marble.marblemain import Brain, Wanderer, expand_wplugins
import marble.plugins  # ensure plugin discovery


class ExpandWPluginsTest(unittest.TestCase):
    def test_expand_shorthand(self) -> None:
        b = Brain(1)
        names = expand_wplugins(["batchtrainer", "*"])
        self.assertIn("batchtrainer", names)
        self.assertGreater(len(names), 1)
        w = Wanderer(b, type_name=",".join(names))
        self.assertGreater(len(getattr(w, "_wplugins", []) or []), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
