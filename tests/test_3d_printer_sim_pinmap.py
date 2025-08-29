import importlib.util
import pathlib
import sys
import unittest

# Load modules from 3d_printer_sim
spec_mc = importlib.util.spec_from_file_location(
    "microcontroller", pathlib.Path("3d_printer_sim/microcontroller.py")
)
mc_module = importlib.util.module_from_spec(spec_mc)
assert spec_mc.loader is not None
sys.modules[spec_mc.name] = mc_module
spec_mc.loader.exec_module(mc_module)
Microcontroller = mc_module.Microcontroller

spec_usb = importlib.util.spec_from_file_location(
    "usb", pathlib.Path("3d_printer_sim/usb.py")
)
usb_module = importlib.util.module_from_spec(spec_usb)
assert spec_usb.loader is not None
sys.modules[spec_usb.name] = usb_module
spec_usb.loader.exec_module(usb_module)
VirtualUSB = usb_module.VirtualUSB

spec_sd = importlib.util.spec_from_file_location(
    "sdcard", pathlib.Path("3d_printer_sim/sdcard.py")
)
sd_module = importlib.util.module_from_spec(spec_sd)
assert spec_sd.loader is not None
sys.modules[spec_sd.name] = sd_module
spec_sd.loader.exec_module(sd_module)
VirtualSDCard = sd_module.VirtualSDCard

spec_sensors = importlib.util.spec_from_file_location(
    "sensors", pathlib.Path("3d_printer_sim/sensors.py")
)
sensors_module = importlib.util.module_from_spec(spec_sensors)
assert spec_sensors.loader is not None
sys.modules[spec_sensors.name] = sensors_module
spec_sensors.loader.exec_module(sensors_module)
TemperatureSensor = sensors_module.TemperatureSensor


class TestPinMapping(unittest.TestCase):
    def test_sensor_auto_maps(self) -> None:
        mc = Microcontroller()
        sensor = TemperatureSensor(mc=mc, pin=0, value=25.0)
        mapped = mc.get_mapped_component(0)
        print("sensor mapped to pin 0:", mapped)
        self.assertIs(mapped, sensor)

    def test_manual_and_interface_mapping(self) -> None:
        mc = Microcontroller()
        mc.map_pin(13, "led")
        print("pin13 mapped:", mc.get_mapped_component(13))
        self.assertEqual(mc.get_mapped_component(13), "led")
        mc.unmap_pin(13)
        print("pin13 after unmap:", mc.get_mapped_component(13))
        self.assertIsNone(mc.get_mapped_component(13))

        usb = VirtualUSB()
        usb.connect()
        mc.attach_usb(usb, pins=[1, 2])
        print("USB mapped pins:", mc.get_mapped_component(1))
        self.assertIs(mc.get_mapped_component(1), usb)

        card = VirtualSDCard()
        card.mount()
        mc.attach_sd_card(card, pins=[10])
        print("SD card mapped pin:", mc.get_mapped_component(10))
        self.assertIs(mc.get_mapped_component(10), card)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
