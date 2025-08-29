import importlib.util
import pathlib
import unittest
import sys

spec_usb = importlib.util.spec_from_file_location(
    "usb", pathlib.Path("3d_printer_sim/usb.py")
)
module_usb = importlib.util.module_from_spec(spec_usb)
assert spec_usb.loader is not None
sys.modules[spec_usb.name] = module_usb
spec_usb.loader.exec_module(module_usb)
VirtualUSB = module_usb.VirtualUSB

spec_mc = importlib.util.spec_from_file_location(
    "microcontroller", pathlib.Path("3d_printer_sim/microcontroller.py")
)
module_mc = importlib.util.module_from_spec(spec_mc)
assert spec_mc.loader is not None
sys.modules[spec_mc.name] = module_mc
spec_mc.loader.exec_module(module_mc)
Microcontroller = module_mc.Microcontroller


class TestVirtualUSB(unittest.TestCase):
    def test_host_device_communication(self):
        usb = VirtualUSB()
        usb.connect()
        usb.send_from_host(b"hi")
        self.assertEqual(usb.read_from_host(), b"hi")
        usb.send_from_device(b"ok")
        self.assertEqual(usb.read_from_device(), b"ok")

    def test_attach_to_microcontroller(self):
        usb = VirtualUSB()
        usb.connect()
        mcu = Microcontroller()
        mcu.attach_usb(usb)
        mcu.usb_send(b"data")
        self.assertEqual(usb.read_from_device(), b"data")
        usb.send_from_host(b"host")
        self.assertEqual(mcu.usb_receive(), b"host")


if __name__ == "__main__":
    unittest.main()
