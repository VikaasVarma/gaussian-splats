"""Persistent Cycles-backed state for the interactive Blender viewer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from gaussian_splats.blender import CAMERA_RENDERERS, CyclesSession, Scene, fit_camera
from gaussian_splats.splats.camera import PinholeCamera

from .controls import CameraController


@dataclass
class BlenderViewer:
    executable: str = "blender"
    session: CyclesSession | None = None
    scene: Scene | None = None
    camera: PinholeCamera | None = None
    controller: CameraController | None = None
    scene_name: str | None = None

    @classmethod
    def create(cls) -> BlenderViewer:
        return cls()

    @property
    def renderers(self) -> tuple[str, ...]:
        return CAMERA_RENDERERS

    def load_scene(self, path: str | Path) -> None:
        self.close()
        self.session = CyclesSession(path, self.executable).__enter__()
        self.scene = self.session.scene
        if self.scene is None:
            raise RuntimeError("Blender did not return scene geometry")
        self.camera = fit_camera(self.scene)
        self.controller = CameraController(self.camera, world_up=(0.0, 0.0, 1.0))
        self.scene_name = Path(path).name

    def update_controls(self, controls: list[dict[str, object]]) -> None:
        if self.controller is None:
            return
        for control in controls:
            self.controller.update(
                control["move"],  # type: ignore[arg-type]
                control["look"],  # type: ignore[arg-type]
                float(control["dt"]),
            )

    def render(self, params: Any) -> torch.Tensor | None:
        if self.session is None or self.camera is None:
            return None
        width = int(params.get("width", self.camera.image_size[0]))
        height = int(params.get("height", self.camera.image_size[1]))
        renderer = params.get("renderer", "cycles")
        return self.session.render(self.camera, width, height, renderer)

    def close(self) -> None:
        if self.session is not None:
            self.session.__exit__(None, None, None)
        self.session = None
        self.scene = None
        self.camera = None
        self.controller = None
