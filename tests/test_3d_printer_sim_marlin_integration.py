import importlib.util
import pathlib
import sys
import tempfile
import unittest

# Load required modules via importlib to keep tests independent
mf_spec = importlib.util.spec_from_file_location(
    "marlin", pathlib.Path("3d_printer_sim/marlin.py")
)
mf_module = importlib.util.module_from_spec(mf_spec)
assert mf_spec.loader is not None
sys.modules[mf_spec.name] = mf_module
mf_spec.loader.exec_module(mf_module)
MarlinFirmware = mf_module.MarlinFirmware

mc_spec = importlib.util.spec_from_file_location(
    "microcontroller", pathlib.Path("3d_printer_sim/microcontroller.py")
)
mc_module = importlib.util.module_from_spec(mc_spec)
assert mc_spec.loader is not None
sys.modules[mc_spec.name] = mc_module
mc_spec.loader.exec_module(mc_module)
Microcontroller = mc_module.Microcontroller

usb_spec = importlib.util.spec_from_file_location(
    "usb", pathlib.Path("3d_printer_sim/usb.py")
)
usb_module = importlib.util.module_from_spec(usb_spec)
assert usb_spec.loader is not None
sys.modules[usb_spec.name] = usb_module
usb_spec.loader.exec_module(usb_module)
VirtualUSB = usb_module.VirtualUSB

sd_spec = importlib.util.spec_from_file_location(
    "sdcard", pathlib.Path("3d_printer_sim/sdcard.py")
)
sd_module = importlib.util.module_from_spec(sd_spec)
assert sd_spec.loader is not None
sys.modules[sd_spec.name] = sd_module
sd_spec.loader.exec_module(sd_module)
VirtualSDCard = sd_module.VirtualSDCard


class TestMarlinIntegration(unittest.TestCase):
    def test_compile_runs_command(self) -> None:
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as build:
            firmware = MarlinFirmware(pathlib.Path(src))
            # Using a no-op command that succeeds
            firmware.compile(pathlib.Path(build), compile_cmd=["true"])

    def test_send_gcode_forwarding(self) -> None:
        mc = Microcontroller()
        usb = VirtualUSB()
        usb.connect()
        mc.attach_usb(usb)
        firmware = MarlinFirmware(pathlib.Path("."))
        firmware.send_gcode(mc, "G28")
        self.assertEqual(usb.host_buffer, [b"G28\n"])

    def test_sd_card_mount(self) -> None:
        mc = Microcontroller()
        card = VirtualSDCard()
        mc.attach_sd_card(card)
        mc.mount_sd_card()
        self.assertTrue(card.mounted)
        mc.unmount_sd_card()
        self.assertFalse(card.mounted)

    def test_feed_sensor_data(self) -> None:
        mc = Microcontroller()
        usb = VirtualUSB()
        usb.connect()
        mc.attach_usb(usb)
        mc.set_digital(1, 1)
        mc.set_analog(2, 3.3)
        firmware = MarlinFirmware(pathlib.Path("."))
        firmware.feed_sensor_data(mc)
        sent = usb.device_buffer[0].decode("ascii")
        self.assertIn("D1:1", sent)
        self.assertIn("A2:3.3", sent)


if __name__ == "__main__":
    unittest.main()

