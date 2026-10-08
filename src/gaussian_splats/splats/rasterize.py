from typing import Literal

import torch

from ..kernels.rasterize.gsplat import rasterize as gsplat_rasterize
from ..kernels.rasterize.interface import rasterize as triton_rasterize
from ..kernels.rasterize.torch import rasterize as torch_rasterize
from .camera import Camera
from .splats import GaussianSplat


def rasterize(
    splats: GaussianSplat,
    camera: Camera,
    tile_size: int = 4,
    opacity_threshold: float = 0.999,
    near: float = 0.2,
    far: float = 100.0,
    rendering_mode: Literal["gaussian", "ellipsoid"] = "gaussian",
    confidence: float = 0.95,
    indices: torch.Tensor | None = None,
    backend: Literal["torch", "triton", "gsplat"] = "torch",
) -> torch.Tensor:
    match backend:
        case "torch":
            rasterize = torch_rasterize
        case "triton":
            rasterize = triton_rasterize
        case "gsplat":
            rasterize = gsplat_rasterize
        case _:
            raise ValueError(f"Unknown rasterization backend: {backend}")
    return rasterize(
        splats,
        camera,
        tile_size,
        opacity_threshold,
        near,
        far,
        rendering_mode,
        confidence,
        indices,
    )


__all__ = ["rasterize"]
