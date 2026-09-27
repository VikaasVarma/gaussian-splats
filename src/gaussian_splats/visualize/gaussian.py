"""Gaussian-specific state and rendering for the browser viewer."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from gaussian_splats.splats.camera import PinholeCamera
from gaussian_splats.splats.rasterize import rasterize
from gaussian_splats.splats.splats import GaussianSplat
from gaussian_splats.timing import record_timings

from .controls import CameraController

CHECKPOINT_ENV = "GAUSSIAN_SPLATS_CHECKPOINT"
DEFAULT_NUM_SPLATS = 10_000


def default_device() -> str:
    forced = os.getenv("GAUSSIAN_SPLATS_DEVICE")
    if forced:
        return forced
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


@dataclass(frozen=True)
class Frame:
    image: torch.Tensor
    timings: dict[str, dict[str, float]]
    metadata: dict[str, str]


@dataclass
class GaussianViewer:
    splats: GaussianSplat | None = None
    camera: PinholeCamera | None = None
    controller: CameraController | None = None
    checkpoint_name: str | None = None

    @classmethod
    def create(cls, checkpoint: str | Path | None = None) -> "GaussianViewer":
        if checkpoint is None:
            return cls()
        device = default_device()
        splats = (
            GaussianSplat.from_checkpoint(checkpoint)
            if checkpoint is not None
            else GaussianSplat(1_000_000)
        ).to(device).eval()
        camera = PinholeCamera().fit_to_points(splats.mean)
        return cls(splats, camera, CameraController(camera), Path(checkpoint).name)

    @classmethod
    def from_environment(cls) -> "GaussianViewer":
        return cls.create(os.getenv(CHECKPOINT_ENV))

    def load_checkpoint(self, checkpoint: str | Path) -> None:
        """Replace the scene and reset the camera to fit the loaded checkpoint."""
        device = self.splats.mean.device if self.splats is not None else default_device()
        splats = GaussianSplat.from_checkpoint(checkpoint).to(device).eval()
        camera = PinholeCamera().fit_to_points(splats.mean)
        self.splats = splats
        self.camera = camera
        self.controller = CameraController(camera)
        self.checkpoint_name = Path(checkpoint).name

    def update_controls(self, controls: Sequence[Mapping[str, object]]) -> None:
        if self.controller is None:
            return
        for control in controls:
            self.controller.update(
                control["move"],  # type: ignore[arg-type]
                control["look"],  # type: ignore[arg-type]
                float(control["dt"]),
            )

    def render(self, params: Mapping[str, str]) -> Frame | None:
        if self.splats is None or self.camera is None:
            return None
        width = int(params.get("width", self.camera.image_size[0]))
        height = int(params.get("height", self.camera.image_size[1]))
        camera = replace(
            self.camera,
            image_size=(width, height),
            principal_point=(width / 2, height / 2),
        )
        num_splats = max(
            1,
            min(int(params.get("num_splats") or DEFAULT_NUM_SPLATS), self.splats.num_points),
        )
        options: dict[str, int | float | str | torch.Tensor] = {
            "tile_size": int(params.get("tile_size", 8)),
            "opacity_threshold": float(params.get("opacity_threshold", 0.999)),
            "near": float(params.get("near", 0.1)),
            "far": float(params.get("far", 100)),
            "rendering_mode": params.get("rendering_mode", "gaussian"),
            "confidence": float(params.get("confidence", 0.95)),
            # Sample across the checkpoint instead of showing only its first rows.
            "indices": torch.linspace(
                0,
                self.splats.num_points - 1,
                num_splats,
                device=self.splats.mean.device,
                dtype=torch.float32,
            ).round().long(),
        }
        times: dict[str, dict[str, float]] = {}
        timing = (
            record_timings(self.splats.mean.device)
            if os.getenv("GAUSSIAN_SPLATS_PROFILE", "1") != "0"
            else nullcontext(times)
        )
        with torch.no_grad(), timing:
            image = rasterize(self.splats, camera, **options)
        return Frame(
            image=image,
            timings=times,
            metadata={
                "X-Total-Splats": str(self.splats.num_points),
                "X-Checkpoint-Name": self.checkpoint_name or "",
            },
        )
