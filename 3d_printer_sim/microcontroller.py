from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
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


@dataclass
class Microcontroller:
    """Simple microcontroller emulator.

    Stores digital and analog pin states in dictionaries and
    provides methods to read and write them. This forms the
    basis for later hardware emulation compatible with Marlin.
    """

    digital_pins: Dict[int, int] = field(default_factory=dict)
    analog_pins: Dict[int, float] = field(default_factory=dict)
    usb: Optional[VirtualUSB] = None

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

    # USB interface helpers
    def attach_usb(self, usb: VirtualUSB) -> None:
        """Attach a :class:`VirtualUSB` interface."""
        self.usb = usb

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
