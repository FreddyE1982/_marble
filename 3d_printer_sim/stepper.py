from __future__ import annotations

"""Simple stepper motor model with kinematic limits.

The model tracks position, velocity and acceleration and exposes a
``set_target_velocity`` method. On each ``update`` call the internal
state advances by ``dt`` seconds while respecting maximum acceleration
and jerk (rate of change of acceleration).

Only the Python standard library is used to keep the module fully
self‑contained inside ``3d_printer_sim``.
"""

from dataclasses import dataclass


@dataclass
class StepperMotor:
    """Represents a single stepper motor axis.

    Parameters
    ----------
    max_acceleration:
        Absolute limit for acceleration in steps/s^2.
    max_jerk:
        Absolute limit for change in acceleration per second
        (steps/s^3).
    """

    max_acceleration: float
    max_jerk: float
    position: float = 0.0
    velocity: float = 0.0
    acceleration: float = 0.0
    target_velocity: float = 0.0

    def set_target_velocity(self, velocity: float) -> None:
        """Set the desired velocity in steps/s."""

        self.target_velocity = float(velocity)

    def update(self, dt: float) -> None:
        """Advance the motor state by ``dt`` seconds.

        The update first adjusts acceleration towards the required
        acceleration to reach ``target_velocity``. The change in
        acceleration is limited by ``max_jerk`` and the resulting
        acceleration is clamped to ``max_acceleration``. Velocity and
        position are then integrated using the updated acceleration.
        """

        if dt <= 0:
            raise ValueError("dt must be positive")

        # Desired acceleration to hit the target velocity in this step
        target_accel = (self.target_velocity - self.velocity) / dt
        delta_accel = target_accel - self.acceleration

        # Limit change in acceleration (jerk)
        max_delta = self.max_jerk * dt
        if delta_accel > max_delta:
            delta_accel = max_delta
        elif delta_accel < -max_delta:
            delta_accel = -max_delta

        self.acceleration += delta_accel

        # Clamp acceleration
        if self.acceleration > self.max_acceleration:
            self.acceleration = self.max_acceleration
        elif self.acceleration < -self.max_acceleration:
            self.acceleration = -self.max_acceleration

        prev_velocity = self.velocity

        # Integrate velocity and position
        self.velocity += self.acceleration * dt

        # Prevent overshoot beyond target velocity
        if self.target_velocity > prev_velocity and self.velocity > self.target_velocity:
            self.velocity = self.target_velocity
            self.acceleration = 0.0
        elif self.target_velocity < prev_velocity and self.velocity < self.target_velocity:
            self.velocity = self.target_velocity
            self.acceleration = 0.0

        self.position += self.velocity * dt


__all__ = ["StepperMotor"]

