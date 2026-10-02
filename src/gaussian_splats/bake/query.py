import torch

from gaussian_splats.splats.splats import GaussianSplat
from gaussian_splats.splats.utils import rgb_to_sh

from .blender import linear_to_srgb
from .renderers.types import RayBackend
from .scene import Scene


def query_scene(
    scene: Scene,
    splats: GaussianSplat,
    backend: RayBackend,
    triangle_id: torch.Tensor,
    barycentric: torch.Tensor,
) -> tuple[GaussianSplat, GaussianSplat]:
    origins = splats.mean
    directions = splats.normals

    rgba, valid = backend.query(
        scene,
        origins,
        directions,
        triangle_ids=triangle_id,
        barycentric=barycentric,
    )

    colors, alphas = rgba[:, :3], rgba[:, 3]
    valid = valid & torch.isfinite(rgba).all(1)
    invalid = ~valid

    display_color = linear_to_srgb(colors[valid])
    color = rgb_to_sh(display_color)[:, None]
    opacity = alphas[valid].clamp(1e-4, 1 - 1e-4).logit()[:, None]

    baked = GaussianSplat.from_tensors(
        splats.mean[valid],
        splats.rotation[valid],
        splats.scale[valid],
        opacity,
        color,
        splats.normals[valid],
    )

    invalid = GaussianSplat.from_tensors(
        splats.mean[invalid],
        splats.rotation[invalid],
        splats.scale[invalid],
        splats.opacity[invalid],
        splats.color[invalid],
        splats.normals[invalid],
    )
    return baked, invalid
