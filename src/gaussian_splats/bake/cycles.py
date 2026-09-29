from __future__ import annotations

import numpy as np
import torch

from gaussian_splats.blender import BlenderSession, Scene


class CyclesBackend:
    def __init__(
        self,
        session: BlenderSession,
        samples: int = 1,
        batch_size: int = 4096,
        eps: float = 1e-4,
    ) -> None:
        self.session = session
        self.samples = samples
        self.batch_size = batch_size
        self.eps = eps

    def query(
        self,
        scene: Scene,
        points: torch.Tensor,
        normals: torch.Tensor,
        *,
        triangle_id: torch.Tensor | None = None,
        barycentric: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device, dtype = points.device, points.dtype
        points_np = points.detach().cpu().numpy()
        normals_np = normals.detach().cpu().numpy()

        scale = float(np.linalg.norm(np.ptp(points_np, axis=0)))
        offset = max(self.eps, scale * 1e-7)
        origins = points_np + normals_np * offset

        output = self.session.request(
            {
                "command": "query",
                "samples": self.samples,
                "batch_size": self.batch_size,
            },
            {
                "origins": origins,
                "directions": normals_np,
                "footprints": np.zeros(len(points_np), dtype=np.float32),
            },
        )
        result = np.load(output)

        return (
            torch.as_tensor(result["colors"], device=device, dtype=dtype),
            torch.as_tensor(result["alphas"], device=device, dtype=dtype),
            torch.as_tensor(result["hits"], device=device, dtype=dtype),
        )
