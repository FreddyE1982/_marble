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

_spec = importlib.util.spec_from_file_location(
    "stepper", pathlib.Path(__file__).with_name("stepper.py")
)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
StepperMotor = _module.StepperMotor


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
    extruded_length: float = 0.0
    deposited_volume: float = 0.0

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
        self.deposited_volume += area * length_mm

    def reset(self) -> None:
        """Reset accumulated extrusion metrics."""

        self.extruded_length = 0.0
        self.deposited_volume = 0.0


__all__ = ["Extruder"]

