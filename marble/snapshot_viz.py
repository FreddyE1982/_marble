"""Utilities for visualizing network topology from snapshots.

This module provides a helper that loads a brain snapshot saved in the
current columnar ``.marble`` format and renders its neuron/synapse topology
to a PNG image. The implementation purposefully stays independent from
``Brain.load_snapshot`` so that visualisation works even in lightweight
environments where the heavy training stack (and torch) is unavailable.
"""

from __future__ import annotations

import math
from array import array
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from .snapshot_stream import SnapshotStreamError, read_latest_state


__all__ = ["snapshot_to_image"]

_TOL = 1e-9


def _load_snapshot_payload(snapshot_path: str) -> Mapping[str, Any]:
    """Load and return the raw snapshot payload dictionary."""

    try:
        data = read_latest_state(snapshot_path)
    except SnapshotStreamError as exc:
        raise ValueError(f"Invalid snapshot stream: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("Snapshot payload must be a mapping")
    return data


def _coerce_sequence(value: Any) -> List[Any]:
    """Convert ``value`` into a list, tolerating ``array`` and numpy inputs."""

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, array):
        return list(value)
    if isinstance(value, (str, bytes)):
        return [value]
    if hasattr(value, "tolist"):
        try:
            converted = value.tolist()
        except Exception:
            pass
        else:
            return list(converted if isinstance(converted, list) else [converted])
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _coords_from_linear_index(linear_index: int, shape: Sequence[int]) -> List[int]:
    dims = [int(max(1, s)) for s in shape]
    if not dims:
        return [int(linear_index)]
    remaining = int(linear_index)
    coords_rev: List[int] = []
    for size_dim in reversed(dims):
        coords_rev.append(remaining % size_dim)
        remaining //= size_dim
    coords = list(reversed(coords_rev))
    if remaining:
        coords[0] += remaining * max(1, dims[0])
    return coords


def _ensure_dims(positions: List[Tuple[float, ...]]) -> List[Tuple[float, ...]]:
    if not positions:
        return positions
    max_dims = max(len(pos) for pos in positions)
    if max_dims == 0:
        return [(0.0,) for _ in positions]
    padded: List[Tuple[float, ...]] = []
    for pos in positions:
        padded_values = [float(pos[i]) if i < len(pos) else 0.0 for i in range(max_dims)]
        padded.append(tuple(padded_values))
    return padded


def _decode_neuron_positions(data: Mapping[str, Any]) -> List[Tuple[float, ...]]:
    neurons_block = data.get("neurons")
    if not isinstance(neurons_block, Mapping):
        positions: List[Tuple[float, ...]] = []
        for entry in _coerce_sequence(neurons_block):
            if isinstance(entry, Mapping):
                coords = entry.get("position") or entry.get("coords")
                if coords is None:
                    continue
                coord_list = [float(v) for v in _coerce_sequence(coords)]
                positions.append(tuple(coord_list))
        return positions

    count_value = neurons_block.get("count")
    count = int(count_value or 0) if count_value is not None else 0
    encoding = str(neurons_block.get("position_encoding", ""))
    dims = int(neurons_block.get("position_dims", data.get("n", 0) or 0))
    if count <= 0 and encoding == "linear":
        count = len(_coerce_sequence(neurons_block.get("linear_indices", [])))
    if count <= 0:
        raw_positions = _coerce_sequence(neurons_block.get("positions", []))
        if dims > 0 and raw_positions:
            count = len(raw_positions) // max(dims, 1)
    if count <= 0:
        weights_values = _coerce_sequence(neurons_block.get("weights", []))
        if weights_values:
            count = len(weights_values)
    if count <= 0:
        return []
    if dims <= 0:
        dims = int(data.get("n", 0) or 0)
    dtype = str(neurons_block.get("position_dtype", "int"))

    positions: List[Tuple[float, ...]] = []
    if encoding == "linear":
        linear_indices = [int(v) for v in _coerce_sequence(neurons_block.get("linear_indices", []))]
        size_values = _coerce_sequence(data.get("size", []))
        size_tuple: Tuple[int, ...]
        if size_values:
            size_tuple = tuple(int(v) for v in size_values)
        else:
            dim_count = dims if dims > 0 else (len(linear_indices) or 1)
            side = int(round(count ** (1.0 / max(1, dim_count)))) or 1
            size_tuple = tuple(side for _ in range(max(1, dim_count)))
        for idx in range(count):
            if idx < len(linear_indices):
                coords = _coords_from_linear_index(linear_indices[idx], size_tuple)
            else:
                coords = [0] * len(size_tuple)
            if dims > 0:
                if len(coords) < dims:
                    coords = coords + [0] * (dims - len(coords))
                elif len(coords) > dims:
                    coords = coords[:dims]
            if dtype == "float":
                positions.append(tuple(float(v) for v in coords))
            else:
                positions.append(tuple(float(int(v)) for v in coords))
        return positions

    raw_positions = _coerce_sequence(neurons_block.get("positions", []))
    if dims <= 0:
        dims = len(raw_positions) // count if count else 0
    if dims <= 0:
        dims = int(data.get("n", 1) or 1)
    for idx in range(count):
        start = idx * dims
        end = start + dims
        chunk = raw_positions[start:end]
        if len(chunk) < dims:
            filler = chunk[-1] if chunk else 0
            chunk = list(chunk) + [filler] * (dims - len(chunk))
        coords = [float(v) for v in chunk]
        if dtype != "float":
            coords = [float(int(c)) for c in coords]
        positions.append(tuple(coords))
    return positions


def _decode_synapse_edges(data: Mapping[str, Any], node_count: int) -> List[Tuple[int, int]]:
    synapses_block = data.get("synapses")
    edges: List[Tuple[int, int]] = []
    if isinstance(synapses_block, Mapping):
        source_list = [int(v) for v in _coerce_sequence(synapses_block.get("source_indices", []))]
        target_list = [int(v) for v in _coerce_sequence(synapses_block.get("target_indices", []))]
        declared = int(synapses_block.get("count", 0))
        edge_count = declared if declared > 0 else min(len(source_list), len(target_list))
        for idx in range(edge_count):
            if idx >= len(source_list) or idx >= len(target_list):
                break
            src = source_list[idx]
            dst = target_list[idx]
            if src < 0 or dst < 0 or src >= node_count or dst >= node_count:
                continue
            edges.append((src, dst))
        return edges

    for entry in _coerce_sequence(synapses_block):
        if not isinstance(entry, Mapping):
            continue
        if "source_idx" in entry and "target_idx" in entry:
            try:
                src = int(entry.get("source_idx"))
                dst = int(entry.get("target_idx"))
            except (TypeError, ValueError):
                continue
        else:
            continue
        if src < 0 or dst < 0 or src >= node_count or dst >= node_count:
            continue
        edges.append((src, dst))
    return edges


def _has_variation(values: Sequence[float]) -> bool:
    if not values:
        return False
    baseline = float(values[0])
    for val in values[1:]:
        if not math.isclose(float(val), baseline, rel_tol=_TOL, abs_tol=_TOL):
            return True
    return False


def _scale_axis(values: Sequence[float], size: int, margin: float) -> List[float]:
    if not values:
        return []
    v_min = min(values)
    v_max = max(values)
    if math.isclose(v_min, v_max, rel_tol=_TOL, abs_tol=_TOL):
        return [size / 2.0 for _ in values]
    span = v_max - v_min
    usable = max(size - 2 * margin, 1.0)
    return [margin + ((val - v_min) / span) * usable for val in values]


def _circle_layout(n: int, size: int) -> Dict[int, Tuple[float, float]]:
    if n <= 0:
        return {}
    center = size / 2.0
    radius = size * 0.4
    return {
        idx: (
            center + radius * math.cos(2 * math.pi * idx / n),
            center + radius * math.sin(2 * math.pi * idx / n),
        )
        for idx in range(n)
    }


def _layout_nodes(positions: List[Tuple[float, ...]], size: int) -> Dict[int, Tuple[float, float]]:
    if not positions:
        return {}
    norm_positions = _ensure_dims(positions)
    dims = len(norm_positions[0]) if norm_positions else 0
    varying_axes: List[int] = []
    for axis in range(dims):
        if _has_variation([pos[axis] for pos in norm_positions]):
            varying_axes.append(axis)

    margin = max(size * 0.08, 8.0)
    if len(varying_axes) >= 2:
        ax_x, ax_y = varying_axes[:2]
        xs = _scale_axis([pos[ax_x] for pos in norm_positions], size, margin)
        ys = _scale_axis([pos[ax_y] for pos in norm_positions], size, margin)
        return {idx: (xs[idx], ys[idx]) for idx in range(len(norm_positions))}

    if len(varying_axes) == 1:
        ax_x = varying_axes[0]
        xs = _scale_axis([pos[ax_x] for pos in norm_positions], size, margin)
        if len(norm_positions) == 1:
            ys = [size / 2.0]
        else:
            usable = max(size - 2 * margin, 1.0)
            step = usable / max(len(norm_positions) - 1, 1)
            ys = [margin + step * idx for idx in range(len(norm_positions))]
        return {idx: (xs[idx], ys[idx]) for idx in range(len(norm_positions))}

    return _circle_layout(len(norm_positions), size)


def snapshot_to_image(snapshot_path: str, output_path: str, *, size: int = 512) -> str:
    """Render a brain snapshot to an image and return the written path.

    Parameters
    ----------
    snapshot_path:
        Path to the ``.marble`` snapshot file to visualise.
    output_path:
        Destination path for the rendered PNG image.
    size:
        Width and height of the generated image in pixels. Defaults to ``512``.

    The function inspects the columnar snapshot payload directly and supports
    both grid and sparse brain snapshots produced by the current
    ``Brain.save_snapshot`` implementation. Neuron coordinates are projected to
    two dimensions using the first varying axes when available; otherwise the
    nodes are arranged on a fallback circle layout.
    """

    payload = _load_snapshot_payload(snapshot_path)
    neuron_positions = _decode_neuron_positions(payload)
    edges = _decode_synapse_edges(payload, len(neuron_positions))

    layout = _layout_nodes(neuron_positions, size)

    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    # Draw edges first so that neuron nodes appear on top.
    for src_idx, dst_idx in edges:
        src = layout.get(src_idx)
        dst = layout.get(dst_idx)
        if src is None or dst is None:
            continue
        draw.line([src, dst], fill="black", width=1)

    # Draw neurons as blue circles.
    node_radius = max(2.0, size * 0.01)
    for idx, (x, y) in layout.items():
        draw.ellipse((x - node_radius, y - node_radius, x + node_radius, y + node_radius), fill="blue")

    img.save(output_path)
    return output_path

