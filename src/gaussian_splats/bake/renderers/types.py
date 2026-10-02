from typing import Protocol

import torch

from gaussian_splats.splats.camera import PinholeCamera

from ..scene import Scene


class RayBackend(Protocol):
    def query(
        self,
        scene: Scene,
        origins: torch.Tensor,  # N x 3
        directions: torch.Tensor,  # N x 3
        **kwargs,
    ) -> tuple[
        torch.Tensor,  # N x 4 (RGBA)
        torch.Tensor,  # N (valid)
    ]: ...


class RenderBackend(Protocol):
    def render(
        self,
        scene: Scene,
        camera: PinholeCamera,
        width: int,
        height: int,
        **kwargs,
    ) -> torch.Tensor:  # H x W x 3 (RGB)
        ...
