from __future__ import annotations

from typing import Protocol

import torch

from gaussian_splats.blender import Scene


class RayQueryBackend(Protocol):
    scene: Scene

    def __enter__(self) -> RayQueryBackend: ...

    def __exit__(self, exception_type, exception, traceback) -> None: ...

    def query(
        self, points: torch.Tensor, normals: torch.Tensor
    ) -> tuple[
        torch.Tensor,  # colors
        torch.Tensor,  # alphas
        torch.Tensor,  # hits
    ]: ...
