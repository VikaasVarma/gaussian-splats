from __future__ import annotations

from typing import Protocol

import torch

from gaussian_splats.blender import Scene


class RayQueryBackend(Protocol):
    def query(
        self,
        scene: Scene,
        points: torch.Tensor,
        normals: torch.Tensor,
        *,
        triangle_id: torch.Tensor | None = None,
        barycentric: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,  # colors
        torch.Tensor,  # alphas
        torch.Tensor,  # hits
    ]: ...
