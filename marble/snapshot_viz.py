"""Utilities for visualizing network topology from snapshots.

This module provides a helper that loads a brain snapshot and
renders its neuron/synapse topology to a simple PNG image.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

from PIL import Image, ImageDraw

from .marblemain import Brain


__all__ = ["snapshot_to_image"]


def snapshot_to_image(snapshot_path: str, output_path: str, *, size: int = 512) -> str:
    """Render a brain snapshot to an image and return the written path.

    Parameters
    ----------
    snapshot_path:
        Path to the ``.marble`` snapshot file to visualize.
    output_path:
        Destination path for the rendered PNG image.
    size:
        Width and height of the generated image in pixels. Defaults to ``512``.

    The function arranges neurons on a circle and draws synapses as straight
    lines between them. It is intended for quick debugging and topology
    inspection rather than publication‑quality figures.
    """

    brain = Brain.load_snapshot(snapshot_path)
    neurons = list(brain.neurons.values())
    n = len(neurons)

    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    if n == 0:
        img.save(output_path)
        return output_path

    center = size / 2
    radius = size * 0.4
    positions: Dict[int, Tuple[float, float]] = {}
    for idx, neuron in enumerate(neurons):
        angle = 2 * math.pi * idx / n
        x = center + radius * math.cos(angle)
        y = center + radius * math.sin(angle)
        positions[id(neuron)] = (x, y)

    # Draw synapses first so neurons overlay them
    for syn in brain.synapses:
        src = positions.get(id(syn.source))
        dst = positions.get(id(syn.target))
        if src and dst:
            draw.line([src, dst], fill="black", width=1)

    # Draw neurons as blue circles
    r = 5
    for x, y in positions.values():
        draw.ellipse((x - r, y - r, x + r, y + r), fill="blue")

    img.save(output_path)
    return output_path

