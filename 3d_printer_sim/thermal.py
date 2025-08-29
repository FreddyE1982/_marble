from __future__ import annotations

"""Thermal component models for hotends and heated beds."""

from dataclasses import dataclass
import importlib.util
import math
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
    heating_rate: float  # coefficient per second while heating
    cooling_rate: float  # coefficient per second while cooling
    temp_range: tuple[float, float] | None = None
    target_temperature: float = 25.0

    def set_target_temperature(self, temp: float) -> None:
        if self.temp_range:
            lo, hi = self.temp_range
            if temp < lo or temp > hi:
                raise ValueError(f"Target temperature {temp} outside allowed range {self.temp_range}")
        self.target_temperature = float(temp)

    def update(self, dt: float) -> None:
        """Advance the thermal simulation by ``dt`` seconds."""

        if dt <= 0:
            raise ValueError("dt must be positive")
        current = self.sensor.read_temperature()
        target = self.target_temperature
        if current < target:
            k = self.heating_rate
        else:
            k = self.cooling_rate
        # Exponential approach to target temperature
        current += (target - current) * (1 - math.exp(-k * dt))
        self.sensor.set_temperature(current)


class Hotend(Heater):
    """Hotend heater."""


class HeatedBed(Heater):
    """Heated bed controller."""


__all__ = ["Heater", "Hotend", "HeatedBed"]

