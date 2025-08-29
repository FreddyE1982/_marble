from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List


def _parse_value(token: str) -> Any:
    token = token.strip()
    if token.lower() in {"true", "false"}:
        return token.lower() == "true"
    try:
        if "." in token:
            return float(token)
        return int(token)
    except ValueError:
        return token


def _parse_yaml(lines: list[str], index: int, indent: int) -> tuple[Any, int]:
    result: dict[str, Any] = {}
    items: list[Any] = []
    is_list = False
    i = index
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        current_indent = len(line) - len(line.lstrip(" "))
        if current_indent < indent:
            break
        stripped = line.strip()
        if stripped.startswith("- "):
            is_list = True
            stripped = stripped[2:]
            i += 1
            # Scalar list item with no following block
            if ":" not in stripped and (i >= len(lines) or len(lines[i]) - len(lines[i].lstrip(" ")) <= current_indent):
                items.append(_parse_value(stripped))
                continue
            item_lines: list[str] = []
            if stripped:
                item_lines.append(" " * (current_indent + 2) + stripped)
            while i < len(lines):
                next_line = lines[i]
                next_indent = len(next_line) - len(next_line.lstrip(" "))
                if next_indent <= current_indent:
                    break
                item_lines.append(next_line)
                i += 1
            value, _ = _parse_yaml(item_lines, 0, current_indent + 2)
            items.append(value)
            continue
        else:
            key, _, rest = stripped.partition(":")
            key = key.strip()
            rest = rest.strip()
            if rest:
                result[key] = _parse_value(rest)
                i += 1
            else:
                i += 1
                value, i = _parse_yaml(lines, i, current_indent + 2)
                result[key] = value
    return (items if is_list else result), i


def parse_simple_yaml(text: str) -> Any:
    lines = [line.rstrip("\n") for line in text.splitlines()]
    data, _ = _parse_yaml(lines, 0, 0)
    return data


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
        data = parse_simple_yaml(fh.read()) or {}
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

