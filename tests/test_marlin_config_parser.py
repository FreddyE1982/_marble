import importlib.util
import pathlib
import sys
import unittest

spec = importlib.util.spec_from_file_location(
    "marlin_config_parser", pathlib.Path("3d_printer_sim/marlin_config_parser.py")
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)

parse_config_items = module.parse_config_items
enumerate_config_items = module.enumerate_config_items


class TestMarlinConfigParser(unittest.TestCase):
    def test_parse_config_items(self) -> None:
        sample = """
#define FOO 1
#define BAR
    #define BAZ 3
"""
        self.assertEqual(parse_config_items(sample), ["FOO", "BAR", "BAZ"])
        print("parsed items:", parse_config_items(sample))

    def test_enumerate_config_items_with_mock(self) -> None:
        module.fetch_marlin_files = lambda files: ["#define ALPHA 1\n", "#define BETA 2\n"]
        items = enumerate_config_items()
        self.assertEqual(items, ["ALPHA", "BETA"])
        print("enumerated items:", items)


if __name__ == "__main__":
    unittest.main()
