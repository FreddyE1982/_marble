from __future__ import annotations

"""Thermal component models for hotends and heated beds."""

from dataclasses import dataclass
import importlib.util
import pathlib
import sys

_spec = importlib.util.spec_from_file_location(
    "sensors", pathlib.Path(__file__).with_name("sensors.py")
)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
TemperatureSensor = _module.TemperatureSensor


@dataclass
class Heater:
    """Base class for temperature controlled components."""

    sensor: TemperatureSensor
    heating_rate: float  # degrees per second while heating
    cooling_rate: float  # degrees per second while cooling
    target_temperature: float = 25.0

    def set_target_temperature(self, temp: float) -> None:
        self.target_temperature = float(temp)

    def update(self, dt: float) -> None:
        """Advance the thermal simulation by ``dt`` seconds."""

        if dt <= 0:
            raise ValueError("dt must be positive")
        current = self.sensor.read_temperature()
        if current < self.target_temperature:
            current += self.heating_rate * dt
            if current > self.target_temperature:
                current = self.target_temperature
        elif current > self.target_temperature:
            current -= self.cooling_rate * dt
            if current < self.target_temperature:
                current = self.target_temperature
        self.sensor.set_temperature(current)


class Hotend(Heater):
    """Hotend heater."""


class HeatedBed(Heater):
    """Heated bed controller."""


__all__ = ["Heater", "Hotend", "HeatedBed"]

