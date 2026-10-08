import warnings
from functools import lru_cache
from math import log1p, sqrt
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange

from ...splats.camera import Camera
from ...splats.splats import GaussianSplat, ProjectedGaussianSplat
from ...splats.utils import (
    evaluate_sh,
    mahalanobis_distance_squared,
    probability,
    quaternion_to_rotation_matrix,
)
from ...timing import timed


@timed
def project(
    splats: GaussianSplat,
    camera: Camera,
    indices: torch.Tensor | None = None,
    covariance_epsilon: float = 0.03,
) -> ProjectedGaussianSplat:
    (mean, rotation, scale, opacity, color, normals) = (
        splats.mean,
        splats.rotation,
        splats.scale,
        splats.opacity,
        splats.color,
        splats.normals,
    )
    if indices is not None:
        mean = mean[indices]
        rotation = rotation[indices]
        scale = scale[indices]
        opacity = opacity[indices]
        color = color[indices]
        normals = normals[indices]

    # Project Gaussian means and covariances.
    mean_world = mean
    mean, covariance = camera.project_gaussian(
        mean,
        quaternion_to_rotation_matrix(F.normalize(rotation, dim=-1)),
        scale.exp(),
        covariance_epsilon=covariance_epsilon,
    )

    # Evaluate spherical harmonics and compute opacity.
    direction = F.normalize(camera.position - mean_world, dim=-1)
    color = evaluate_sh(color, direction)
    opacity = opacity.sigmoid()
    normals = normals @ camera.rotation_matrix.T

    return ProjectedGaussianSplat(mean, covariance, color, opacity, normals)


@timed
def cull(
    splats: ProjectedGaussianSplat, near: float = 0.1, far: float = 100.0
) -> ProjectedGaussianSplat:
    visible = (splats.mean[:, 2] > near) & (splats.mean[:, 2] < far)
    return ProjectedGaussianSplat(
        splats.mean[visible],
        splats.covariance[visible],
        splats.color[visible],
        splats.opacity[visible],
        None if splats.normals is None else splats.normals[visible],
    )


@timed
def tile(
    splats: ProjectedGaussianSplat,
    camera: Camera,
    tile_size: int = 8,
    rendering_mode: Literal["gaussian", "ellipsoid"] = "gaussian",
    confidence: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert tile_size > 0 and tile_size & (tile_size - 1) == 0
    W, H = camera.image_size
    assert W % tile_size == 0 and H % tile_size == 0
    Ht, Wt = H // tile_size, W // tile_size

    if rendering_mode == "gaussian":
        extent = (2 * torch.log(splats.opacity.squeeze(-1) * 255).sqrt().clamp_min(0))[:, None]
    elif rendering_mode == "ellipsoid":
        extent = sqrt(-2 * log1p(-confidence))
    else:
        raise ValueError(f"Invalid rendering mode: {rendering_mode}")

    radius = extent * splats.covariance.diagonal(dim1=-2, dim2=-1).sqrt()

    # Estimate tiles that Gaussians intersect
    bounds = torch.stack((splats.mean[:, :2] - radius, splats.mean[:, :2] + radius), dim=-1)
    bounds = (bounds / tile_size).floor().int()
    bounds[:, :, 1] += 1  # N x 2 x 2 (N, x/y, min/max)

    limits = bounds.new_tensor((Wt, Ht))[:, None]
    bounds.clamp_(torch.zeros_like(limits), limits)

    tile_min, tile_max = bounds[:, :, 0], bounds[:, :, 1]
    size = tile_max - tile_min
    counts = size.prod(dim=-1)

    # Assign Gaussian IDs to tiles
    gaussian_ids = torch.repeat_interleave(
        torch.arange(len(bounds), device=bounds.device, dtype=torch.int32), counts
    )

    # Get corresponding tile IDs
    starts = counts.cumsum(0) - counts
    local_id = torch.arange(counts.sum(), device=splats.mean.device)
    local_id -= torch.repeat_interleave(starts, counts)
    origins = tile_min[gaussian_ids]
    widths = size[gaussian_ids, 0]
    tile_ids = (origins[:, 1] + local_id // widths) * Wt + origins[:, 0] + local_id % widths

    # Order by tile ID x depth
    depths = splats.mean[gaussian_ids, 2]
    if depths.dtype != torch.float32:
        warnings.warn(
            f"Converting intersection depths from {depths.dtype} to float32 for key packing.",
            RuntimeWarning,
            stacklevel=2,
        )
        depths = depths.float()
    depth_bits = depths.contiguous().view(torch.int32).to(torch.int64)
    keys = (tile_ids.to(torch.int64) << 32) | (depth_bits & 0xFFFFFFFF)
    keys, order = keys.sort()
    gaussian_ids = gaussian_ids[order]
    tile_ids = keys >> 32

    # Count Gaussians per tile
    counts = torch.bincount(tile_ids, minlength=Ht * Wt)
    ends = counts.cumsum(0)
    ranges = torch.stack((ends - counts, ends), -1)

    return gaussian_ids, ranges


@lru_cache
def create_pixel_grids_and_origins(
    tile_size: int,
    tiles_h: int,
    tiles_w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[
    torch.Tensor,  # Pixel Coordinates: tile_size x tile_size x 2
    torch.Tensor,  # Tile Origins: tiles_h x tiles_w x 2
]:
    y, x = torch.meshgrid(
        torch.arange(tile_size, device=device, dtype=dtype),
        torch.arange(tile_size, device=device, dtype=dtype),
        indexing="ij",
    )
    tile_y, tile_x = torch.meshgrid(
        torch.arange(tiles_h, device=device, dtype=dtype),
        torch.arange(tiles_w, device=device, dtype=dtype),
        indexing="ij",
    )
    tile_pixels = rearrange(torch.stack((x, y), -1), "h w c -> (h w) c")
    tile_origins = tile_size * rearrange(torch.stack((tile_x, tile_y), -1), "h w c -> (h w) c")
    return tile_pixels, tile_origins


@timed
def render(
    splats: ProjectedGaussianSplat,
    camera: Camera,
    assignment: tuple[torch.Tensor, torch.Tensor],
    tile_size: int = 8,
    opacity_threshold: float = 0.999,
    rendering_mode: Literal["gaussian", "ellipsoid"] = "gaussian",
    confidence: float = 0.95,
) -> torch.Tensor:
    assert tile_size > 0 and tile_size & (tile_size - 1) == 0
    (
        mean,
        inverse_covariance,
        color,
        opacity,
    ) = (splats.mean, splats.inverse_covariance, splats.color, splats.opacity)
    indices, ranges = assignment

    W, H = camera.image_size
    assert W % tile_size == 0 and H % tile_size == 0
    Ht, Wt = H // tile_size, W // tile_size
    counts = ranges[:, 1] - ranges[:, 0]

    # Get the pixel grid for each tile.
    tile_ids = torch.repeat_interleave(torch.arange(Ht * Wt, device=mean.device), counts)
    tile_pixels, tile_origins = create_pixel_grids_and_origins(
        tile_size, Ht, Wt, mean.device, mean.dtype
    )

    # Compute the probability of each Gaussian at each pixel.
    offset = tile_origins[tile_ids, None] + tile_pixels - mean[indices, None, :2]
    distance = mahalanobis_distance_squared(offset, inverse_covariance[indices])
    alpha = (opacity[indices] * probability(distance, rendering_mode, confidence)).clamp_max(0.99)
    alpha = torch.where(alpha >= 1.0 / 255.0, alpha, 0.0)

    # Accumulate transmittance and color contributions.
    log_survival = torch.log1p(-alpha)
    transmittance = log_survival.cumsum(0, dtype=torch.float64) - log_survival
    transmittance = torch.exp(transmittance - transmittance[ranges[tile_ids, 0]])
    transmittance = transmittance.to(alpha.dtype)

    weights = transmittance * alpha * ((1 - transmittance) < opacity_threshold)
    contributions = weights[..., None] * color[indices, None]
    image = contributions.new_zeros(Ht * Wt, tile_size * tile_size, 3)
    image = image.index_add(0, tile_ids, contributions)

    return rearrange(
        image,
        "(ht wt) (th tw) c -> (ht th) (wt tw) c",
        ht=Ht,
        wt=Wt,
        th=tile_size,
    )[:H, :W]


def rasterize(
    splats: GaussianSplat,
    camera: Camera,
    tile_size: int = 8,
    opacity_threshold: float = 0.999,
    near: float = 1e-4,
    far: float = 100.0,
    rendering_mode: Literal["gaussian", "ellipsoid"] = "gaussian",
    confidence: float = 0.95,
    indices: torch.Tensor | None = None,
    covariance_epsilon: float = 0.03,
) -> torch.Tensor:
    projected = project(splats, camera, indices, covariance_epsilon=covariance_epsilon)
    projected = cull(projected, near, far)
    assignment = tile(projected, camera, tile_size, rendering_mode, confidence)
    image = render(
        projected, camera, assignment, tile_size, opacity_threshold, rendering_mode, confidence
    )

    return image
