from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import importlib.util
import pathlib
import sys

_spec = importlib.util.spec_from_file_location(
    "usb", pathlib.Path(__file__).with_name("usb.py")
)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
VirtualUSB = _module.VirtualUSB

_sd_spec = importlib.util.spec_from_file_location(
    "sdcard", pathlib.Path(__file__).with_name("sdcard.py")
)
_sd_module = importlib.util.module_from_spec(_sd_spec)
assert _sd_spec.loader is not None
sys.modules[_sd_spec.name] = _sd_module
_sd_spec.loader.exec_module(_sd_module)
VirtualSDCard = _sd_module.VirtualSDCard


@dataclass
class Microcontroller:
    """Simple microcontroller emulator.

    Stores digital and analog pin states in dictionaries and
    provides methods to read and write them. This forms the
    basis for later hardware emulation compatible with Marlin.
    """

    digital_pins: Dict[int, int] = field(default_factory=dict)
    analog_pins: Dict[int, float] = field(default_factory=dict)
    pin_mapping: Dict[int, Any] = field(default_factory=dict)
    usb: Optional[VirtualUSB] = None
    sd_card: Optional[VirtualSDCard] = None

    def set_digital(self, pin: int, value: int) -> None:
        if value not in (0, 1):
            raise ValueError("Digital pin value must be 0 or 1")
        self.digital_pins[pin] = value

    def read_digital(self, pin: int) -> int:
        return self.digital_pins.get(pin, 0)

    def set_analog(self, pin: int, value: float) -> None:
        self.analog_pins[pin] = float(value)

    def read_analog(self, pin: int) -> float:
        return self.analog_pins.get(pin, 0.0)

    # Pin mapping helpers
    def map_pin(self, pin: int, component: Any) -> None:
        """Associate a component with a microcontroller pin."""
        self.pin_mapping[pin] = component

    def get_mapped_component(self, pin: int) -> Any | None:
        """Return the component mapped to *pin*, if any."""
        return self.pin_mapping.get(pin)

    def unmap_pin(self, pin: int) -> None:
        self.pin_mapping.pop(pin, None)

    # USB interface helpers
    def attach_usb(self, usb: VirtualUSB, pins: Optional[list[int]] = None) -> None:
        """Attach a :class:`VirtualUSB` interface and optionally map pins."""
        self.usb = usb
        if pins:
            for pin in pins:
                self.map_pin(pin, usb)

    def detach_usb(self) -> None:
        self.usb = None

    def usb_send(self, data: bytes) -> None:
        if not self.usb:
            raise RuntimeError("USB not attached")
        self.usb.send_from_device(data)

    def usb_receive(self) -> bytes | None:
        if not self.usb:
            raise RuntimeError("USB not attached")
        return self.usb.read_from_host()

    # SD card helpers
    def attach_sd_card(self, card: VirtualSDCard, pins: Optional[list[int]] = None) -> None:
        """Attach a :class:`VirtualSDCard` and optionally map pins."""
        self.sd_card = card
        if pins:
            for pin in pins:
                self.map_pin(pin, card)

    def detach_sd_card(self) -> None:
        self.sd_card = None

    def sd_write_file(self, path: str, data: bytes) -> None:
        if not self.sd_card:
            raise RuntimeError("SD card not attached")
        self.sd_card.write_file(path, data)

    def sd_read_file(self, path: str) -> bytes | None:
        if not self.sd_card:
            raise RuntimeError("SD card not attached")
        return self.sd_card.read_file(path)

    def sd_list_files(self) -> list[str]:
        if not self.sd_card:
            raise RuntimeError("SD card not attached")
        return self.sd_card.list_files()
