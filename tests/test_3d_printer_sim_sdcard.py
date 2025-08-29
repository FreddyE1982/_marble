import importlib.util
import pathlib
import sys
import unittest

spec_sd = importlib.util.spec_from_file_location(
    "sdcard", pathlib.Path("3d_printer_sim/sdcard.py")
)
module_sd = importlib.util.module_from_spec(spec_sd)
assert spec_sd.loader is not None
sys.modules[spec_sd.name] = module_sd
spec_sd.loader.exec_module(module_sd)
VirtualSDCard = module_sd.VirtualSDCard

spec_mc = importlib.util.spec_from_file_location(
    "microcontroller", pathlib.Path("3d_printer_sim/microcontroller.py")
)
module_mc = importlib.util.module_from_spec(spec_mc)
assert spec_mc.loader is not None
sys.modules[spec_mc.name] = module_mc
spec_mc.loader.exec_module(module_mc)
Microcontroller = module_mc.Microcontroller


class TestSDCard(unittest.TestCase):
    def test_basic_file_ops(self) -> None:
        card = VirtualSDCard()
        card.mount()
        card.write_file("test.gcode", b"G1 X10")
        print("files after write:", card.list_files())
        self.assertEqual(card.read_file("test.gcode"), b"G1 X10")
        self.assertEqual(card.list_files(), ["test.gcode"])
        card.delete_file("test.gcode")
        print("files after delete:", card.list_files())
        self.assertEqual(card.list_files(), [])

    def test_microcontroller_sd_interface(self) -> None:
        mc = Microcontroller()
        card = VirtualSDCard()
        card.mount()
        mc.attach_sd_card(card)
        mc.sd_write_file("a.txt", b"abc")
        print("mc sd files:", mc.sd_list_files())
        self.assertEqual(mc.sd_read_file("a.txt"), b"abc")
        self.assertEqual(mc.sd_list_files(), ["a.txt"])


if __name__ == "__main__":
    unittest.main()
