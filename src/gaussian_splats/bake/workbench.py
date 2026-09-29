from __future__ import annotations

import torch

from gaussian_splats.blender import Scene
from gaussian_splats.workbench import query_radiance


class WorkbenchBackend:
    def __init__(
        self,
        eps: float = 1e-4,
    ) -> None:
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
        assert triangle_id is not None and barycentric is not None, (
            "WorkbenchBackend requires surface provenance"
        )
        return query_radiance(
            scene,
            points,
            normals,
            triangle_id,
            barycentric,
            eps=self.eps,
        )
