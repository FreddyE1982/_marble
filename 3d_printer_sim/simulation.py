"""High-level printer simulation tying physics to visualization.

The :class:`PrinterSimulation` class instantiates simple stepper-based
axes, an :class:`Extruder`, and a :class:`PrinterVisualizer`.  Updating
the simulation advances the physics models and keeps the visual
representation in sync, depositing filament segments whenever material
is extruded.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import pathlib
import sys
import math

# Local imports using importlib to keep the module self-contained
_stepper_spec = importlib.util.spec_from_file_location(
    "stepper", pathlib.Path(__file__).with_name("stepper.py")
)
_stepper_module = importlib.util.module_from_spec(_stepper_spec)
sys.modules[_stepper_spec.name] = _stepper_module
assert _stepper_spec.loader is not None
_stepper_spec.loader.exec_module(_stepper_module)
StepperMotor = _stepper_module.StepperMotor

_extruder_spec = importlib.util.spec_from_file_location(
    "extruder", pathlib.Path(__file__).with_name("extruder.py")
)
_extruder_module = importlib.util.module_from_spec(_extruder_spec)
sys.modules[_extruder_spec.name] = _extruder_module
assert _extruder_spec.loader is not None
_extruder_spec.loader.exec_module(_extruder_module)
Extruder = _extruder_module.Extruder

_viz_spec = importlib.util.spec_from_file_location(
    "visualization", pathlib.Path(__file__).with_name("visualization.py")
)
_viz_module = importlib.util.module_from_spec(_viz_spec)
sys.modules[_viz_spec.name] = _viz_module
assert _viz_spec.loader is not None
_viz_spec.loader.exec_module(_viz_module)
PrinterVisualizer = _viz_module.PrinterVisualizer


@dataclass(init=False)
class PrinterSimulation:
    """Combine motion, extrusion and visualization."""

    visualizer: PrinterVisualizer
    x_motor: StepperMotor
    y_motor: StepperMotor
    z_motor: StepperMotor
    extruder: Extruder
    _last_extruded: float = 0.0

    bed_tilt_x: float
    bed_tilt_y: float
    gravity: tuple[float, float, float]

    # Nozzle state relative to the bed
    nozzle_diameter: float
    layer_height: float
    nozzle_height: float
    extrusion_width: float
    adhesion: float
    collision: bool
    surface_damage: float
    part_detached: bool
    layer_height_error: float
    bed_temperature: float
    optimal_bed_temp: float

    def __init__(self, bed_tilt_x: float = 0.0, bed_tilt_y: float = 0.0) -> None:
        self.bed_tilt_x = float(bed_tilt_x)
        self.bed_tilt_y = float(bed_tilt_y)
        self.gravity = self._compute_gravity()
        self.visualizer = PrinterVisualizer()
        self.visualizer.set_bed_tilt(self.bed_tilt_x, self.bed_tilt_y)
        self.x_motor = StepperMotor(max_acceleration=1000, max_jerk=1000)
        self.y_motor = StepperMotor(max_acceleration=1000, max_jerk=1000)
        self.z_motor = StepperMotor(max_acceleration=1000, max_jerk=1000)
        e_motor = StepperMotor(max_acceleration=1000, max_jerk=1000)
        self.extruder = Extruder(e_motor, steps_per_mm=100, filament_diameter=1.75)
        self._last_extruded = 0.0

        self.nozzle_diameter = 0.4
        self.layer_height = 0.2
        self.nozzle_height = 0.0
        self.extrusion_width = self.nozzle_diameter
        self.adhesion = 1.0
        self.collision = False
        self.surface_damage = 0.0
        self.part_detached = False
        self.layer_height_error = 0.0
        self.bed_temperature = 60.0
        self.optimal_bed_temp = 60.0

    # -------------------------------------------------------------- controls
    def set_axis_velocities(self, vx: float, vy: float, vz: float) -> None:
        """Set target velocities for the three motion axes."""

        self.x_motor.set_target_velocity(vx)
        self.y_motor.set_target_velocity(vy)
        self.z_motor.set_target_velocity(vz)

    def set_extrusion_velocity(self, velocity: float) -> None:
        """Set filament feed velocity for the extruder."""

        self.extruder.set_target_velocity(velocity)

    # ---------------------------------------------------------------- update
    def update(self, dt: float) -> None:
        """Advance physics models and update the visualization."""

        self.x_motor.update(dt)
        self.y_motor.update(dt)
        self.z_motor.update(dt)
        self.extruder.update(dt)

        x, y, z = self._apply_tilt(
            self.x_motor.position, self.y_motor.position, self.z_motor.position
        )

        # --- nozzle to bed distance and related effects ---
        self.nozzle_height = z
        self.layer_height_error = self.nozzle_height - self.layer_height
        if self.nozzle_height < 0:
            self.collision = True
            self.surface_damage += -self.nozzle_height
            if abs(self.x_motor.velocity) + abs(self.y_motor.velocity) > 1:
                self.part_detached = True
            base_adh = 0.0
            width = self.nozzle_diameter
        else:
            self.collision = False
            layer_z = round(self.nozzle_height / self.layer_height) * self.layer_height
            diff = self.nozzle_height - layer_z
            if abs(diff) >= self.layer_height * 0.5:
                base_adh = 0.0
            else:
                base_adh = 1 - abs(diff) / (self.layer_height * 0.5)
            if len(self.visualizer.filament) == 0:
                if self.nozzle_height < self.layer_height:
                    squish = (self.layer_height - self.nozzle_height) / self.layer_height
                    width = self.nozzle_diameter * (1 + squish)
                else:
                    width = self.nozzle_diameter
                temp_factor = max(
                    0.0,
                    1.0
                    - abs(self.bed_temperature - self.optimal_bed_temp)
                    / self.optimal_bed_temp,
                )
                base_adh *= temp_factor
            else:
                width = self.nozzle_diameter

        flow = self.extruder.last_flow_efficiency
        self.adhesion = base_adh * flow
        if self.adhesion < 1e-12:
            self.adhesion = 0.0
        self.extrusion_width = width

        self.visualizer.update_extruder_position(x, y, z)

        if self.extruder.extruded_length > self._last_extruded:
            if self.adhesion > 0.5 and not self.collision and not self.part_detached:
                deposit_z = max(z, 0.0)
                self.visualizer.add_filament_segment(
                    x, y, deposit_z, radius=self.extrusion_width / 2
                )
            self._last_extruded = self.extruder.extruded_length

    # ------------------------------------------------------------ internals
    def _apply_tilt(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        rx = math.radians(self.bed_tilt_x)
        ry = math.radians(self.bed_tilt_y)
        # rotate around X
        y1 = y * math.cos(rx) - z * math.sin(rx)
        z1 = y * math.sin(rx) + z * math.cos(rx)
        # rotate around Y
        x2 = x * math.cos(ry) + z1 * math.sin(ry)
        z2 = -x * math.sin(ry) + z1 * math.cos(ry)
        return x2, y1, z2

    def _compute_gravity(self) -> tuple[float, float, float]:
        rx = math.radians(self.bed_tilt_x)
        ry = math.radians(self.bed_tilt_y)
        gx, gy, gz = 0.0, 0.0, -9.81
        # undo rotation around Y then X to express gravity in bed coordinates
        gx, gz = gx * math.cos(-ry) - gz * math.sin(-ry), gx * math.sin(-ry) + gz * math.cos(-ry)
        gy, gz = gy * math.cos(-rx) - gz * math.sin(-rx), gy * math.sin(-rx) + gz * math.cos(-rx)
        return (gx, gy, gz)


__all__ = ["PrinterSimulation"]
