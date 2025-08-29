"""3D visualization module for the printer simulator.

This module uses ``pythreejs`` to render a basic printer model.  It
provides multiple camera views and updates to reflect simulated motion and
extrusion.  All geometry lives in this file so the simulator remains
self‑contained apart from the external dependency on ``pythreejs``.
"""

from __future__ import annotations

from typing import List

from pythreejs import (
    AmbientLight,
    BoxGeometry,
    CylinderGeometry,
    LineSegments,
    LineBasicMaterial,
    Mesh,
    MeshLambertMaterial,
    PerspectiveCamera,
    Renderer,
    Scene,
    OrbitControls,
)


class PrinterVisualizer:
    """Render and update a virtual 3D printer."""

    def __init__(self) -> None:
        self.scene = Scene()

        # Lighting
        self.scene.add(AmbientLight(color="#ffffff"))

        # Bed
        bed_geom = BoxGeometry(200, 2, 200)
        bed_mat = MeshLambertMaterial(color="#808080")
        self.bed = Mesh(geometry=bed_geom, material=bed_mat)
        self.scene.add(self.bed)

        # Extruder block
        ext_geom = BoxGeometry(10, 10, 10)
        ext_mat = MeshLambertMaterial(color="#ff0000")
        self.extruder = Mesh(geometry=ext_geom, material=ext_mat)
        self.extruder.position = (0, 10, 0)
        self.scene.add(self.extruder)

        # Mechanical axes represented as simple rails
        rail_mat = MeshLambertMaterial(color="#404040")

        # X axis rail
        x_geom = BoxGeometry(200, 2, 2)
        self.x_rail = Mesh(geometry=x_geom, material=rail_mat)
        self.x_rail.position = (0, 11, 0)
        self.scene.add(self.x_rail)

        # Y axis rail
        y_geom = BoxGeometry(2, 2, 200)
        self.y_rail = Mesh(geometry=y_geom, material=rail_mat)
        self.y_rail.position = (0, 11, 0)
        self.scene.add(self.y_rail)

        # Z axis rail
        z_geom = BoxGeometry(2, 200, 2)
        self.z_rail = Mesh(geometry=z_geom, material=rail_mat)
        self.z_rail.position = (0, 100, 0)
        self.scene.add(self.z_rail)

        # Axis lines for reference
        axis_mat = LineBasicMaterial(color="#0000ff")
        axis_geom = BoxGeometry(0, 0, 0)
        self.axes = LineSegments(geometry=axis_geom, material=axis_mat)
        self.scene.add(self.axes)

        # Filament deposits
        self.filament: List[Mesh] = []

        # Camera and renderer
        self.camera = PerspectiveCamera(position=[200, 200, 200], up=[0, 1, 0])
        self.camera.lookAt((0, 0, 0))
        self.controls = OrbitControls(controlling=self.camera)
        self.renderer = Renderer(camera=self.camera, scene=self.scene, controls=[self.controls])

    # ------------------------------------------------------------------ views
    def _set_camera(self, position: List[float]) -> Renderer:
        self.camera.position = position
        self.camera.lookAt((0, 0, 0))
        return self.renderer

    def get_renderer(self) -> Renderer:
        """Return the renderer with interactive camera controls."""
        return self.renderer

    def get_isometric_view(self) -> Renderer:
        """Return renderer set to isometric view."""
        return self._set_camera([200, 200, 200])

    def get_side_view(self) -> Renderer:
        """Return renderer set to side view."""
        return self._set_camera([200, 0, 0])

    def get_top_view(self) -> Renderer:
        """Return renderer set to top-down view."""
        return self._set_camera([0, 200, 0])

    # --------------------------------------------------------------- simulation
    def update_extruder_position(self, x: float, y: float, z: float) -> None:
        self.extruder.position = (x, y, z)

    def add_filament_segment(self, x: float, y: float, z: float, radius: float = 0.5) -> None:
        cyl = CylinderGeometry(radiusTop=radius, radiusBottom=radius, height=1)
        mat = MeshLambertMaterial(color="#ffa500")
        seg = Mesh(cyl, mat)
        seg.position = (x, y, z)
        self.scene.add(seg)
        self.filament.append(seg)

    def set_bed_tilt(self, x_deg: float, y_deg: float) -> None:
        """Rotate the virtual bed around the X and Y axes."""

        from math import radians

        try:
            self.bed.rotation = (radians(x_deg), radians(y_deg), 0, "XYZ")
        except Exception:
            pass


__all__ = ["PrinterVisualizer"]

