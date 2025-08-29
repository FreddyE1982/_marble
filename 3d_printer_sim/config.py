from __future__ import annotations

from dataclasses import dataclass
from typing import List
import yaml


@dataclass
class Volume:
    x: float
    y: float
    z: float

    @staticmethod
    def from_dict(d: dict, name: str) -> "Volume":
        for axis in ("x", "y", "z"):
            if axis not in d:
                raise ValueError(f"{name} missing axis {axis}")
            if d[axis] <= 0:
                raise ValueError(f"{name}.{axis} must be positive")
        return Volume(float(d["x"]), float(d["y"]), float(d["z"]))


@dataclass
class BedSize:
    x: float
    y: float

    @staticmethod
    def from_dict(d: dict) -> "BedSize":
        for axis in ("x", "y"):
            if axis not in d:
                raise ValueError(f"bed_size missing axis {axis}")
            if d[axis] <= 0:
                raise ValueError(f"bed_size.{axis} must be positive")
        return BedSize(float(d["x"]), float(d["y"]))


@dataclass
class Extruder:
    id: int
    type: str
    hotend: str

    @staticmethod
    def from_dict(d: dict) -> "Extruder":
        if "id" not in d or "type" not in d or "hotend" not in d:
            raise ValueError("extruder requires id, type and hotend")
        return Extruder(int(d["id"]), str(d["type"]), str(d["hotend"]))


@dataclass
class PrinterConfig:
    build_volume: Volume
    bed_size: BedSize
    max_print_dimensions: Volume
    extruders: List[Extruder]


def load_config(path: str) -> PrinterConfig:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if "build_volume" not in data or "bed_size" not in data or "max_print_dimensions" not in data or "extruders" not in data:
        raise ValueError("config missing required sections")
    build_volume = Volume.from_dict(data["build_volume"], "build_volume")
    bed_size = BedSize.from_dict(data["bed_size"])
    max_print = Volume.from_dict(data["max_print_dimensions"], "max_print_dimensions")
    if not isinstance(data["extruders"], list) or not data["extruders"]:
        raise ValueError("extruders must be a non-empty list")
    extruders = [Extruder.from_dict(e) for e in data["extruders"]]
    return PrinterConfig(
        build_volume=build_volume,
        bed_size=bed_size,
        max_print_dimensions=max_print,
        extruders=extruders,
    )

