"""Integration helpers for running the Marlin firmware."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import pathlib
import subprocess
import sys

# Local imports via importlib to avoid package-level coupling
_mc_spec = importlib.util.spec_from_file_location(
    "microcontroller", pathlib.Path(__file__).with_name("microcontroller.py")
)
_mc_module = importlib.util.module_from_spec(_mc_spec)
assert _mc_spec.loader is not None
sys.modules[_mc_spec.name] = _mc_module
_mc_spec.loader.exec_module(_mc_module)
Microcontroller = _mc_module.Microcontroller


@dataclass
class MarlinFirmware:
    """Handle compilation and interaction with Marlin firmware."""

    source_dir: pathlib.Path

    def compile(self, build_dir: pathlib.Path, compile_cmd: list[str] | None = None) -> None:
        """Compile the firmware using *compile_cmd* in *source_dir*.

        Parameters
        ----------
        build_dir:
            Directory where the build output is placed.
        compile_cmd:
            Command list to invoke the build system. Defaults to
            ``["platformio", "run"]``.
        """

        build_dir.mkdir(parents=True, exist_ok=True)
        if compile_cmd is None:
            compile_cmd = ["platformio", "run"]
        subprocess.run(compile_cmd, cwd=self.source_dir, check=True)

    def send_gcode(self, controller: Microcontroller, command: str) -> None:
        """Forward *command* to Marlin via the controller's USB link."""

        controller.send_gcode(command)

    def feed_sensor_data(self, controller: Microcontroller) -> None:
        """Transmit current sensor readings to Marlin."""

        controller.transmit_sensor_data()


__all__ = ["MarlinFirmware"]

