from typing import Literal

import torch

from ...splats.camera import Camera
from ...splats.splats import GaussianSplat

try:
    from gsplat import rasterization
except ImportError as error:

    def rasterization(*args, error=error, **kwargs):
        raise RuntimeError("The gsplat backend requires the 'gsplat' package") from error


def _camera_matrices(camera: Camera) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert the project camera convention to gsplat's camera convention."""
    rotation = camera.rotation_matrix
    translation = camera.translation
    viewmat = torch.eye(4, dtype=rotation.dtype, device=rotation.device)
    viewmat[:3, :3] = rotation
    viewmat[:3, 3] = translation

    # Translate our camera (facing -Z with +Y up) to
    # gsplat's camera (facing +Z with -Y up).
    frame = torch.diag(rotation.new_tensor((1.0, -1.0, -1.0)))
    viewmat[:3] = frame @ viewmat[:3]

    fx, fy = camera.focal_length
    cx, cy = camera.principal_point
    intrinsics = rotation.new_tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return viewmat, intrinsics


def rasterize(
    splats: GaussianSplat,
    camera: Camera,
    tile_size: int = 16,
    opacity_threshold: float = 0.999,
    near: float = 0.2,
    far: float = 100.0,
    rendering_mode: Literal["gaussian", "ellipsoid"] = "gaussian",
    confidence: float = 0.95,
    indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Render splats with gsplat while preserving the local rasterizer API."""
    if rendering_mode != "gaussian":
        raise ValueError("gsplat only supports Gaussian rendering")
    if opacity_threshold != 0.999:
        raise ValueError("gsplat does not support opacity_threshold")
    if confidence != 0.95:
        raise ValueError("gsplat does not support confidence")

    mean, rotation, scale, opacity, color = (
        splats.mean,
        splats.rotation,
        splats.scale,
        splats.opacity,
        splats.color,
    )
    if indices is not None:
        mean, rotation, scale, opacity, color = (
            mean[indices],
            rotation[indices],
            scale[indices],
            opacity[indices],
            color[indices],
        )

    viewmat, intrinsics = _camera_matrices(camera)
    width, height = camera.image_size
    sh_degree = int(color.shape[1] ** 0.5) - 1
    backgrounds = color.new_zeros(3)
    rendered, _, _ = rasterization(
        means=mean,
        quats=rotation,
        scales=scale.exp(),
        opacities=opacity.sigmoid().squeeze(-1),
        colors=color,
        viewmats=viewmat[None],
        Ks=intrinsics[None],
        width=width,
        height=height,
        near_plane=near,
        far_plane=far,
        eps2d=0.3,
        sh_degree=sh_degree,
        packed=True,
        tile_size=tile_size,
        backgrounds=backgrounds,
    )
    return rendered[0]
