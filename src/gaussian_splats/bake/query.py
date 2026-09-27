import torch

from gaussian_splats.splats.splats import GaussianSplat
from gaussian_splats.splats.utils import rgb_to_sh

from .backend import RayQueryBackend


def _linear_to_srgb(color: torch.Tensor) -> torch.Tensor:
    return torch.where(
        color <= 0.0031308,
        color * 12.92,
        1.055 * color.clamp_min(0).pow(1 / 2.4) - 0.055,
    )


def query_scene(
    splats: GaussianSplat,
    backend: RayQueryBackend,
    eps: float = 1e-4,
) -> tuple[GaussianSplat, GaussianSplat]:
    positions = splats.mean
    normals = splats.normals
    extent = torch.linalg.vector_norm(positions.amax(dim=0) - positions.amin(dim=0))
    eps = max(float(eps), float(extent) * 1e-7)

    colors, alphas, hits = backend.query(positions, normals)

    distance = torch.norm(hits - positions, dim=1)
    valid = torch.isfinite(alphas) & torch.isfinite(hits).all(1) & (distance <= eps * 2)
    invalid = ~valid

    display_color = _linear_to_srgb(colors[valid])
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
