from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from gaussian_splats.splats.camera import PinholeCamera

from .scene import Lights, Materials, Scene

CAMERA_RENDERERS = ("cycles", "eevee", "workbench")
RENDERERS = CAMERA_RENDERERS


def fit_camera(scene: Scene) -> PinholeCamera:
    points = scene.vertices.reshape(-1, 3)
    rotation = torch.tensor(
        [math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0],
        dtype=points.dtype,
        device=points.device,
    )
    camera = PinholeCamera(rotation=rotation).fit_to_points(points)
    position = camera.position.clone()
    position[2] = 0.5 * (points[:, 2].amax() + points[:, 2].amin())
    camera.translation.copy_(-(camera.rotation_matrix @ position))
    return camera


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
        worker = Path(__file__).with_name("blender_worker.py")
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
            path=self.mesh,
            vertices=torch.from_numpy(data["positions"]),
            normals=torch.from_numpy(data["normals"]),
            mesh_ids=torch.from_numpy(data["mesh_id"]),
            uvs=torch.from_numpy(data["uvs"]),
            material_ids=torch.from_numpy(data["material_id"]),
            materials=Materials(
                colors=torch.from_numpy(data["material_color"]),
                speculars=torch.from_numpy(data["material_specular"]),
                shininesses=torch.from_numpy(data["material_shininess"]),
                texture_rects=torch.from_numpy(data["texture_rect"]),
                texture_atlas=torch.from_numpy(data["texture_atlas"]),
            ),
            background_color=torch.from_numpy(data["background_color"]),
            lights=Lights(
                positions=torch.from_numpy(data["light_positions"]),
                directions=torch.from_numpy(data["light_directions"]),
                colors=torch.from_numpy(data["light_colors"]),
                energies=torch.from_numpy(data["light_energies"]),
                types=torch.from_numpy(data["light_types"]),
            ),
        )
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        if self._process is not None:
            if self._process.poll() is None and self._process.stdin is not None:
                try:
                    self._process.stdin.write(json.dumps({"command": "close"}) + "\n")
                    self._process.stdin.flush()
                except BrokenPipeError:
                    pass  # Preserve the worker's original failure, if any.
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
        if command["command"] == "query":
            batches = math.ceil(len(arrays["origins"]) / command["batch_size"])
            with tqdm(total=batches, desc="Cycles queries", unit="batch") as progress:
                output = self._read_until(marker, progress)
        else:
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

    def _read_until(self, marker: str, progress=None) -> str | None:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("BlenderSession is not running")
        logs = []
        for line in self._process.stdout:
            if line.startswith("CYCLES_DEVICE") or (progress is not None and progress.n == 0):
                tqdm.write(line.rstrip())
            if line.startswith("QUERY_BATCH_DONE") and progress is not None:
                progress.update(1)
                continue
            if line.startswith(marker):
                return None
            logs.append(line)
            if self._process.poll() is not None:
                break
        return "".join(logs)

    def render_reference(
        self,
        camera: PinholeCamera,
        destination: str | Path,
        *,
        samples: int = 4096,
        seed: int = 0,
        exposure: float = 4.0,
        near: float = 1e-4,
        far: float = 100.0,
        device: str = "CPU",
    ) -> Path:
        """Persist EXR, 16-bit display PNG, and float/mask bundle outside session tempdir."""
        if samples < 1 or seed < 0:
            raise ValueError("Samples must be positive and seed nonnegative")
        width, height = camera.image_size
        if camera.principal_point != (width / 2, height / 2):
            raise ValueError("Reference camera currently requires centered principal point")
        if abs(camera.focal_length[0] - camera.focal_length[1]) > 1e-6:
            raise ValueError("Reference camera currently requires square pixels")
        output = self.request(
            {
                "command": "render",
                "reference_bundle": True,
                "renderer": "cycles",
                "samples": samples,
                "seed": seed,
                "exposure": exposure,
                "device": device,
                "width": width,
                "height": height,
                "near": near,
                "far": far,
                "position": camera.position.tolist(),
                "rotation": camera.rotation_matrix.tolist(),
                "focal_length": list(camera.focal_length),
            },
            output_suffix=".npz",
            marker="RENDER_RESULT",
        )
        destination = Path(destination).with_suffix(".npz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".npz", ".exr", ".png"):
            shutil.copy2(output.with_suffix(suffix), destination.with_suffix(suffix))
        return destination

    def render(
        self,
        camera: PinholeCamera,
        width: int,
        height: int,
        renderer: str = "cycles",
        samples: int = 1,
        exposure: float = 0.0,
    ) -> torch.Tensor:
        output = self.request(
            {
                "command": "render",
                "renderer": renderer,
                "samples": samples,
                "exposure": exposure,
                "width": width,
                "height": height,
                "position": camera.position.tolist(),
                "rotation": camera.rotation_matrix.tolist(),
                "focal_length": list(camera.focal_length),
            },
            output_suffix=".png",
            marker="RENDER_RESULT",
        )
        image = np.array(Image.open(output).convert("RGB"), copy=True)
        return torch.from_numpy(image).float() / 255


def linear_to_srgb(color: torch.Tensor) -> torch.Tensor:
    return torch.where(
        color <= 0.0031308,
        color * 12.92,
        1.055 * color.clamp_min(0).pow(1 / 2.4) - 0.055,
    )


def srgb_to_linear(color: torch.Tensor) -> torch.Tensor:
    return torch.where(
        color <= 0.04045,
        color / 12.92,
        ((color.clamp_min(0) + 0.055) / 1.055).pow(2.4),
    )
