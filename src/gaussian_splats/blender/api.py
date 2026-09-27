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
from PIL import Image

from gaussian_splats.splats.camera import PinholeCamera

CAMERA_RENDERERS = ("cycles", "eevee", "workbench")
RENDERERS = CAMERA_RENDERERS


@dataclass(frozen=True)
class Scene:
    path: Path
    vertices: torch.Tensor  # T x 3 x 3
    normals: torch.Tensor  # T x 3 x 3
    mesh_id: torch.Tensor  # T

    def to(self, device: torch.device) -> Scene:
        return Scene(
            self.path,
            self.vertices.to(device),
            self.normals.to(device),
            self.mesh_id.to(device),
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


class CyclesSession:
    def __init__(
        self,
        mesh: str | Path,
        executable: str | Path = "blender",
        samples: int = 1,
        batch_size: int = 4096,
    ) -> None:
        self.mesh = Path(mesh)
        self.executable = executable
        self.samples = samples
        self.batch_size = batch_size
        self.scene: Scene | None = None
        self._process: subprocess.Popen[str] | None = None
        self._directory: tempfile.TemporaryDirectory[str] | None = None
        self._request = 0

    def __enter__(self) -> CyclesSession:
        if not self.mesh.is_file():
            raise FileNotFoundError(self.mesh)
        if not 1 <= self.batch_size <= 4096:
            raise ValueError("batch_size must be between 1 and 4096")
        executable = shutil.which(str(self.executable)) or str(self.executable)
        worker = Path(__file__).with_name("worker.py")
        self._directory = tempfile.TemporaryDirectory(prefix="gaussian-splats-cycles-")
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
                str(self.samples),
                str(self.batch_size),
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
            raise RuntimeError(f"Blender Cycles session failed to start:\n{output or ''}")
        data = np.load(scene_path)
        self.scene = Scene(
            self.mesh,
            torch.from_numpy(data["positions"]),
            torch.from_numpy(data["normals"]),
            torch.from_numpy(data["mesh_id"]),
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

    def query(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        footprints: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        input_path, output_path = self._request_paths("query", ".result.npz")
        np.savez(input_path, origins=origins, directions=directions, footprints=footprints)
        self._send({"command": "query", "input": str(input_path), "output": str(output_path)})
        output = self._read_until("RESULT")
        if output is not None:
            raise RuntimeError(f"Blender Cycles query failed:\n{output}")
        result = np.load(output_path)
        return result["colors"], result["alphas"], result["hits"]

    def render_camera(self, camera: PinholeCamera, width: int, height: int) -> torch.Tensor:
        return self.render(camera, width, height, "cycles")

    def render(
        self,
        camera: PinholeCamera,
        width: int,
        height: int,
        renderer: str = "cycles",
    ) -> torch.Tensor:
        if renderer not in CAMERA_RENDERERS:
            raise ValueError(f"Unsupported camera renderer: {renderer}")
        _, output_path = self._request_paths("render", ".png")
        request = {
            "command": "render",
            "renderer": renderer,
            "output": str(output_path),
            "width": width,
            "height": height,
            "position": camera.position.tolist(),
            "rotation": camera.rotation_matrix.tolist(),
            "focal_length": list(camera.focal_length),
        }
        self._send(request)
        output = self._read_until("RENDER_RESULT")
        if output is not None:
            raise RuntimeError(f"Blender Cycles render failed:\n{output}")
        image = np.array(Image.open(output_path).convert("RGB"), copy=True)
        return torch.from_numpy(image).float() / 255

    def _request_paths(self, kind: str, suffix: str = ".npz") -> tuple[Path, Path]:
        if self._directory is None:
            raise RuntimeError("CyclesSession must be entered before querying")
        request = Path(self._directory.name) / f"request-{self._request:05d}-{kind}"
        self._request += 1
        return request.with_suffix(".input.npz"), request.with_suffix(suffix)

    def _send(self, request: dict[str, object]) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("CyclesSession must be entered before sending requests")
        self._process.stdin.write(json.dumps(request) + "\n")
        self._process.stdin.flush()

    def _read_until(self, marker: str) -> str | None:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("CyclesSession is not running")
        logs = []
        for line in self._process.stdout:
            if line.startswith(marker):
                return None
            logs.append(line)
            if self._process.poll() is not None:
                break
        return "".join(logs)


def query_rays(
    mesh: str | Path,
    origins: np.ndarray,
    directions: np.ndarray,
    footprints: np.ndarray,
    executable: str | Path = "blender",
    samples: int = 1,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with CyclesSession(mesh, executable, samples, batch_size) as session:
        return session.query(origins, directions, footprints)
