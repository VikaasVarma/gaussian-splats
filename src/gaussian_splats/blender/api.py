from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import torch

from gaussian_splats.splats.camera import PinholeCamera

CAMERA_RENDERERS = ("cycles", "eevee", "workbench")
RENDERERS = CAMERA_RENDERERS


@dataclass(frozen=True)
class Scene:
    path: Path
    vertices: torch.Tensor  # T x 3 x 3
    normals: torch.Tensor  # T x 3 x 3
    mesh_id: torch.Tensor  # T
    uvs: torch.Tensor  # T x 3 x 2
    material_id: torch.Tensor  # T
    material_color: torch.Tensor  # M x 4
    texture_rect: torch.Tensor  # M x 4: x, y, width, height
    texture_atlas: torch.Tensor  # H x W x 4

    def to(self, device: torch.device) -> Scene:
        return Scene(
            self.path,
            self.vertices.to(device),
            self.normals.to(device),
            self.mesh_id.to(device),
            self.uvs.to(device),
            self.material_id.to(device),
            self.material_color.to(device),
            self.texture_rect.to(device),
            self.texture_atlas.to(device),
        )

    @cached_property
    def mesh_bounds(self) -> torch.Tensor:  # n_meshes x 2 x 3
        n_meshes = int(self.mesh_id.max()) + 1
        points = self.vertices.view(-1, 3)
        index = self.mesh_id.view(-1, 1, 1).expand_as(self.vertices).view(-1, 3)
        minimum = points.new_full((n_meshes, 3), float("inf"))
        maximum = points.new_full((n_meshes, 3), -float("inf"))
        minimum.scatter_reduce_(0, index, points, reduce="amin")
        maximum.scatter_reduce_(0, index, points, reduce="amax")
        return torch.stack((minimum, maximum), dim=1)

    @cached_property
    def scene_bounds(self) -> torch.Tensor:  # 2 x 3
        points = self.vertices.view(-1, 3)
        return torch.stack((points.amin(0), points.amax(0)))

    @cached_property
    def surface_areas(self) -> torch.Tensor:  # T
        return torch.linalg.norm(
            torch.linalg.cross(
                self.vertices[:, 1] - self.vertices[:, 0],
                self.vertices[:, 2] - self.vertices[:, 0],
            ),
            dim=-1,
        ).mul(0.5)


def fit_camera(scene: Scene) -> PinholeCamera:
    # Transform Blnder (Z-up) to Ours (Y-up)
    points = scene.vertices.reshape(-1, 3)
    rotation = torch.tensor([0.0, 0.0, 2**-0.5, 2**-0.5], dtype=points.dtype, device=points.device)
    return PinholeCamera(rotation=rotation).fit_to_points(points)


class BlenderSession:
    def __init__(
        self,
        mesh: str | Path,
        executable: str | Path = "blender",
    ) -> None:
        self.mesh = Path(mesh)
        self.executable = executable
        self.scene: Scene | None = None
        self._process: subprocess.Popen[str] | None = None
        self._directory: tempfile.TemporaryDirectory[str] | None = None
        self._request = 0

    def __enter__(self) -> BlenderSession:
        if not self.mesh.is_file():
            raise FileNotFoundError(self.mesh)
        executable = shutil.which(str(self.executable)) or str(self.executable)
        worker = Path(__file__).with_name("worker.py")
        self._directory = tempfile.TemporaryDirectory(prefix="gaussian-splats-blender-")
        directory = Path(self._directory.name)
        scene_path = directory / "scene.npz"
        self._process = subprocess.Popen(
            [
                executable,
                "-b",
                "--python",
                str(worker),
                "--",
                str(self.mesh.resolve()),
                str(scene_path),
                "--server",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output = self._read_until("READY")
        if output is not None or not scene_path.is_file():
            self.__exit__(None, None, None)
            raise RuntimeError(f"Blender session failed to start:\n{output or ''}")
        data = np.load(scene_path)
        self.scene = Scene(
            self.mesh,
            torch.from_numpy(data["positions"]),
            torch.from_numpy(data["normals"]),
            torch.from_numpy(data["mesh_id"]),
            torch.from_numpy(data["uvs"]),
            torch.from_numpy(data["material_id"]),
            torch.from_numpy(data["material_color"]),
            torch.from_numpy(data["texture_rect"]),
            torch.from_numpy(data["texture_atlas"]),
        )
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        if self._process is not None:
            if self._process.poll() is None and self._process.stdin is not None:
                self._process.stdin.write(json.dumps({"command": "close"}) + "\n")
                self._process.stdin.flush()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        if self._directory is not None:
            self._directory.cleanup()
        self._process = None
        self._directory = None
        self.scene = None

    def request(
        self,
        command: dict[str, object],
        arrays: dict[str, np.ndarray] | None = None,
        output_suffix: str = ".result.npz",
        marker: str = "RESULT",
    ) -> Path:
        input_path, output_path = self._request_paths(command["command"], output_suffix)
        if arrays is not None:
            np.savez(input_path, **arrays)
            command = {**command, "input": str(input_path)}
        self._send({**command, "output": str(output_path)})
        output = self._read_until(marker)
        if output is not None:
            raise RuntimeError(f"Blender request failed:\n{output}")
        return output_path

    def _request_paths(self, kind: str, suffix: str = ".npz") -> tuple[Path, Path]:
        if self._directory is None:
            raise RuntimeError("BlenderSession must be entered before querying")
        request = Path(self._directory.name) / f"request-{self._request:05d}-{kind}"
        self._request += 1
        return request.with_suffix(".input.npz"), request.with_suffix(suffix)

    def _send(self, request: dict[str, object]) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("BlenderSession must be entered before sending requests")
        self._process.stdin.write(json.dumps(request) + "\n")
        self._process.stdin.flush()

    def _read_until(self, marker: str) -> str | None:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("BlenderSession is not running")
        logs = []
        for line in self._process.stdout:
            if line.startswith(marker):
                return None
            logs.append(line)
            if self._process.poll() is not None:
                break
        return "".join(logs)
