from __future__ import annotations

import numpy as np
import torch

from gaussian_splats.splats.camera import PinholeCamera

from ..blender import BlenderSession
from ..scene import Scene
from .types import RayBackend, RenderBackend


class CyclesBackend(RayBackend, RenderBackend):
    def __init__(
        self,
        session: BlenderSession,
        samples: int = 1,
        batch_size: int = 4096,
        eps: float = 1e-4,
        exposure: float = 0.0,
        device: str = "CPU",
    ) -> None:
        self.session = session
        self.samples = samples
        self.batch_size = batch_size
        self.eps = eps
        self.exposure = exposure
        self.device = device

    def query(
        self,
        scene: Scene,
        origins: torch.Tensor,
        directions: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device, dtype = origins.device, origins.dtype
        origins_np = origins.detach().cpu().numpy()
        directions_np = directions.detach().cpu().numpy()

        scale = float(np.linalg.norm(np.ptp(origins_np, axis=0)))
        offset = max(self.eps, scale * 1e-7)
        ray_origins = origins_np + directions_np * offset

        output = self.session.request(
            {
                "command": "query",
                "samples": self.samples,
                "batch_size": self.batch_size,
                "device": self.device,
            },
            {
                "origins": ray_origins,
                "directions": directions_np,
                "footprints": np.zeros(len(origins_np), dtype=np.float32),
            },
        )
        result = np.load(output)

        colors, alphas, hits = map(
            lambda key: torch.as_tensor(result[key], device=device, dtype=dtype),
            ["colors", "alphas", "hits"],
        )
        colors = colors * 2**self.exposure

        valid = torch.isfinite(hits).all(1) & (
            torch.linalg.vector_norm(hits - origins, dim=1) <= self.eps * 2
        )
        return torch.cat((colors, alphas[:, None]), dim=1), valid

    def render(
        self,
        scene: Scene,
        camera: PinholeCamera,
        width: int,
        height: int,
        **kwargs: object,
    ) -> torch.Tensor:
        return self.session.render(
            camera,
            width,
            height,
            renderer="cycles",
            samples=self.samples,
            exposure=float(kwargs.get("exposure", 0.0)),
        )
