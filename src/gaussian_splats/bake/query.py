import math

import torch
import torch.nn.functional as F

from gaussian_splats.splats.splats import GaussianSplat
from gaussian_splats.splats.utils import spherical_harmonics

from .blender import linear_to_srgb
from .renderers.types import RayBackend
from .scene import Scene


def query_scene(
    scene: Scene,
    splats: GaussianSplat,
    backend: RayBackend,
    triangle_id: torch.Tensor,
    barycentric: torch.Tensor,
    sh_degree: int = 3,
    view_samples: int = 32,
    sh_smoothing_percent: float = 0.0,
) -> tuple[GaussianSplat, GaussianSplat]:
    directions = sample_outgoing_directions(splats.normals, view_samples)
    rgba, valid = backend.query(
        scene,
        splats.mean.repeat_interleave(view_samples, dim=0),
        directions.flatten(0, 1),
        triangle_ids=triangle_id.repeat_interleave(view_samples),
        barycentric=barycentric.repeat_interleave(view_samples, dim=0),
    )

    valid = valid.reshape(-1, view_samples)
    rgba = rgba.reshape(-1, view_samples, 4).masked_fill(~valid[..., None], 0)
    counts = valid.sum(1)
    keep = counts > 0
    print(
        f"Invalid queries: {(~valid).sum().item()}/{valid.numel()} "
        f"({(~valid).float().mean().item():.2%}); "
        f"invalid splats: {(~keep).sum().item()}/{len(keep)}; "
        f"partially invalid: {((counts > 0) & (counts < view_samples)).sum().item()}"
    )

    # Fit SH coefficients to valid queries
    color = fit_sh(rgba[..., :3], directions, valid, sh_degree)[keep]

    # Average opacities over valid queries
    opacity = rgba[keep, :, 3].sum(1) / counts[keep]
    opacity = opacity.clamp(1e-4, 1 - 1e-4).logit()[:, None]

    if sh_smoothing_percent:  # Smooth SH coeefficients across mesh
        _, groups, sizes = torch.unique(triangle_id[keep], return_inverse=True, return_counts=True)
        means = color.new_zeros(len(sizes), *color.shape[1:]).index_add_(0, groups, color)
        means /= sizes[:, None, None]
        color = torch.lerp(color, means[groups], sh_smoothing_percent / 100)

    baked = GaussianSplat.from_tensors(
        splats.mean[keep],
        splats.rotation[keep],
        splats.scale[keep],
        opacity,
        color,
        splats.normals[keep],
    )

    invalid = GaussianSplat.from_tensors(
        splats.mean[~keep],
        splats.rotation[~keep],
        splats.scale[~keep],
        splats.opacity[~keep],
        splats.color[~keep],
        splats.normals[~keep],
    )
    return baked, invalid


def sample_outgoing_directions(normals: torch.Tensor, count: int) -> torch.Tensor:
    """Sample random directions in hemisphere around surface normals"""
    if count == 1:
        return normals[:, None]

    device, dtype = normals.device, normals.dtype
    index = torch.arange(count, device=device, dtype=dtype)
    z = (index + 0.5) / count
    phi = 2 * math.pi * torch.frac(index / ((1 + 5**0.5) / 2))
    radius = torch.sqrt((1 - z.square()).clamp_min(0))
    local = torch.stack((radius * phi.cos(), radius * phi.sin(), z), dim=-1)

    reference = normals.new_tensor((0.0, 0.0, 1.0)).expand_as(normals)
    reference = torch.where(
        normals[:, 2:3].abs() > 0.9, normals.new_tensor((0.0, 1.0, 0.0)), reference
    )
    tangent = F.normalize(torch.linalg.cross(reference, normals), dim=-1)
    bitangent = torch.linalg.cross(normals, tangent)
    return (
        local[None, :, 0:1] * tangent[:, None]
        + local[None, :, 1:2] * bitangent[:, None]
        + local[None, :, 2:3] * normals[:, None]
    )


def fit_sh(
    colors: torch.Tensor,  # N x V x 3
    directions: torch.Tensor,  # N x V x 3
    valid: torch.Tensor,  # N x V
    degree: int,
) -> torch.Tensor:  # N x (degree + 1)^2 x 3
    N, V, _ = directions.shape
    C = (degree + 1) ** 2

    # Evaluate SH basis at each query direction
    basis = spherical_harmonics(directions.reshape(-1, 3), C).reshape(N, V, C)

    # Convert radiance to display color and remove the SH renderer's offset
    colors = colors.masked_fill(~valid[..., None], 0).clamp_min(0)
    target = linear_to_srgb(colors).clamp(0, 1) - 0.5

    # Build normal equations using valid queries
    weights = valid[..., None].to(colors.dtype)
    weighted_basis = (basis * weights).mT  # N x C x V
    gram = weighted_basis @ basis  # N x C x C
    rhs = weighted_basis @ target  # N x C x 3

    # Regualirize higher order coefficients
    bands = torch.arange(C, device=colors.device).float().sqrt().floor()
    regularization = (1 + bands).square() * 1e-3
    gram = gram + torch.diag(regularization.to(colors.dtype))

    return torch.linalg.solve(gram, rhs)
