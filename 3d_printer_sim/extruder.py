from __future__ import annotations

"""Extrusion and material deposition model.

The :class:`Extruder` couples a stepper motor with basic filament
geometry to track how much material has been extruded.  It remains
self‑contained inside ``3d_printer_sim`` and only relies on modules in
this directory.
"""

from dataclasses import dataclass
import importlib.util
import math
import pathlib
import sys


def _load_local(name: str):
    """Load a sibling module by *name* using :mod:`importlib`.

    This keeps the ``3d_printer_sim`` package self‑contained without
    relying on package imports that might escape the directory.
    """

    spec = importlib.util.spec_from_file_location(
        name, pathlib.Path(__file__).with_name(f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


StepperMotor = _load_local("stepper").StepperMotor
TemperatureSensor = _load_local("sensors").TemperatureSensor


@dataclass
class Extruder:
    """Represents a filament extruder.

    Parameters
    ----------
    stepper:
        Stepper motor driving the filament.
    steps_per_mm:
        Number of motor steps that move one millimetre of filament.
    filament_diameter:
        Diameter of the filament in millimetres used to compute volume.
    """

    stepper: StepperMotor
    steps_per_mm: float
    filament_diameter: float
    temperature_sensor: TemperatureSensor | None = None
    melt_temperature: float = 200.0
    viscosity_factor: float = 0.1
    extruded_length: float = 0.0
    deposited_volume: float = 0.0
    last_flow_efficiency: float = 1.0

    def set_target_velocity(self, velocity: float) -> None:
        """Proxy to the stepper's ``set_target_velocity``."""

        self.stepper.set_target_velocity(velocity)

    def update(self, dt: float) -> None:
        """Advance simulation by ``dt`` seconds.

        The stepper state is updated and the resulting change in
        position is converted into extruded filament length and volume.
        """

        prev_pos = self.stepper.position
        self.stepper.update(dt)
        delta_steps = self.stepper.position - prev_pos
        length_mm = delta_steps / self.steps_per_mm
        if length_mm <= 0:
            return

        self.extruded_length += length_mm

        area = math.pi * (self.filament_diameter / 2) ** 2
        if self.temperature_sensor is not None:
            temp = self.temperature_sensor.read_temperature()
            eff = 1.0 / (
                1.0 + math.exp(-self.viscosity_factor * (temp - self.melt_temperature))
            )
        else:
            eff = 1.0
        self.last_flow_efficiency = eff
        self.deposited_volume += area * length_mm * eff

    def reset(self) -> None:
        """Reset accumulated extrusion metrics."""

        self.extruded_length = 0.0
        self.deposited_volume = 0.0
        self.last_flow_efficiency = 1.0


__all__ = ["Extruder"]

