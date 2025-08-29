from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AnalogSensor:
    """Basic analog sensor updating a microcontroller pin."""

    mc: Any
    pin: int
    value: float = 0.0

    def __post_init__(self) -> None:
        self.mc.set_analog(self.pin, self.value)
        if hasattr(self.mc, "map_pin"):
            self.mc.map_pin(self.pin, self)

    def set_value(self, value: float) -> None:
        self.value = float(value)
        self.mc.set_analog(self.pin, self.value)

    def read_value(self) -> float:
        return self.value


@dataclass
class TemperatureSensor(AnalogSensor):
    """Temperature sensor that stores degrees Celsius."""

    def set_temperature(self, temp: float) -> None:
        self.set_value(temp)

    def read_temperature(self) -> float:
        return self.read_value()
