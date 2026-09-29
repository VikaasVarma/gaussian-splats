import torch

from gaussian_splats.blender import Scene, linear_to_srgb
from gaussian_splats.splats.splats import GaussianSplat
from gaussian_splats.splats.utils import rgb_to_sh

from .backend import RayQueryBackend


def query_scene(
    scene: Scene,
    splats: GaussianSplat,
    backend: RayQueryBackend,
    triangle_id: torch.Tensor,
    barycentric: torch.Tensor,
    eps: float = 1e-4,
) -> tuple[GaussianSplat, GaussianSplat]:
    positions = splats.mean
    normals = splats.normals
    extent = torch.linalg.vector_norm(positions.amax(dim=0) - positions.amin(dim=0))
    eps = max(float(eps), float(extent) * 1e-7)

    colors, alphas, hits = backend.query(
        scene,
        positions,
        normals,
        triangle_id=triangle_id,
        barycentric=barycentric,
    )

    distance = torch.norm(hits - positions, dim=1)
    valid = torch.isfinite(alphas) & torch.isfinite(hits).all(1) & (distance <= eps * 2)
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
