from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Microcontroller:
    """Simple microcontroller emulator.

    Stores digital and analog pin states in dictionaries and
    provides methods to read and write them. This forms the
    basis for later hardware emulation compatible with Marlin.
    """

    digital_pins: Dict[int, int] = field(default_factory=dict)
    analog_pins: Dict[int, float] = field(default_factory=dict)

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
