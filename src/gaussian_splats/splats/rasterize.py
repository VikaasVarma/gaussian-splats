import math
from functools import lru_cache
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from ..timing import timed
from .camera import Camera
from .splats import GaussianSplat, ProjectedGaussianSplat
from .utils import (
    evaluate_sh,
    mahalanobis_distance_squared,
    probability,
    quaternion_to_rotation_matrix,
)


@timed
def project(
    splats: GaussianSplat, camera: Camera, count: int | None = None
) -> ProjectedGaussianSplat:
    count = count or len(splats.mean)

    # Project gaussian means and covariances
    mean_world = splats.mean[:count]
    mean, covariance = camera.project_gaussian(
        mean_world,
        quaternion_to_rotation_matrix(F.normalize(splats.rotation[:count], dim=-1)),
        splats.scale[:count].exp(),
    )

    # Evaluate SH and compute opacity
    direction = F.normalize(mean_world - camera.position, dim=-1)
    color = evaluate_sh(splats.color[:count], direction)
    opacity = splats.opacity[:count].sigmoid()

    return ProjectedGaussianSplat(
        mean=mean,
        covariance=covariance,
        color=color,
        opacity=opacity,
    )


@timed
def cull(
    splats: ProjectedGaussianSplat, near: float = 0.1, far: float = 100.0
) -> ProjectedGaussianSplat:
    visible = (splats.mean[:, 2] > near) & (splats.mean[:, 2] < far)

    return ProjectedGaussianSplat(
        mean=splats.mean[visible],
        covariance=splats.covariance[visible],
        color=splats.color[visible],
        opacity=splats.opacity[visible],
    )


@timed
def tile(
    splats: ProjectedGaussianSplat,
    camera: Camera,
    tile_size: int = 8,
    rendering_mode: str = "gaussian",
    confidence: float = 0.95,
) -> tuple[
    torch.Tensor,  # Gaussian IDs: M ordered by tile ID x depth
    torch.Tensor,  # Tile ranges: (Ht * Wt) x 2 ∈ [0, M)
]:
    W, H = camera.image_size
    Ht, Wt = H // tile_size, W // tile_size

    # Estimate Gaussian extents
    extent = math.sqrt(-2 * math.log1p(-confidence)) if rendering_mode == "ellipsoid" else 3
    radius = extent * splats.covariance.diagonal(dim1=-2, dim2=-1).clamp_min(0).sqrt()

    # Estimate tiles that Gaussians intersect
    bounds = torch.stack((splats.mean[:, :2] - radius, splats.mean[:, :2] + radius))  # 2 x N x 2
    bounds = (bounds / tile_size).floor().long()
    bounds[1] += 1

    limits = bounds.new_tensor((Wt, Ht))
    bounds.clamp_(torch.zeros_like(limits), limits)

    tile_min, tile_max = bounds.unbind()
    size = tile_max - tile_min
    counts = size.prod(dim=-1)

    # Assign Gaussian IDs to tiles
    gaussian_ids = torch.repeat_interleave(  # M
        torch.arange(len(splats.mean), device=splats.mean.device), counts
    )

    # Get corresponding tile IDs
    starts = counts.cumsum(0) - counts
    local_id = torch.arange(counts.sum(), device=splats.mean.device)
    local_id -= torch.repeat_interleave(starts, counts)
    origins = tile_min[gaussian_ids]
    widths = size[gaussian_ids, 0]
    tile_ids = (origins[:, 1] + local_id // widths) * Wt + origins[:, 0] + local_id % widths

    # Order by tile ID x depth
    order = splats.mean[gaussian_ids, 2].argsort(stable=True)
    order = order[tile_ids[order].argsort(stable=True)]
    gaussian_ids, tile_ids = gaussian_ids[order], tile_ids[order]

    # Count Gaussians per tile
    counts = torch.bincount(tile_ids, minlength=Ht * Wt)
    ends = counts.cumsum(0)
    ranges = torch.stack((ends - counts, ends), -1)

    return gaussian_ids, ranges


@lru_cache
def create_pixel_grids_and_origins(
    tile_size: int, Ht: int, Wt: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    y, x = torch.meshgrid(
        torch.arange(tile_size, device=device, dtype=dtype),
        torch.arange(tile_size, device=device, dtype=dtype),
        indexing="ij",
    )
    tile_y, tile_x = torch.meshgrid(
        torch.arange(Ht, device=device, dtype=dtype),
        torch.arange(Wt, device=device, dtype=dtype),
        indexing="ij",
    )

    tile_pixels = rearrange(torch.stack((x, y), -1), "h w c -> (h w) c")
    tile_origins = tile_size * rearrange(torch.stack((tile_x, tile_y), -1), "h w c -> (h w) c")
    return tile_pixels, tile_origins


def render(
    splats: ProjectedGaussianSplat,
    camera: Camera,
    assignment: tuple[torch.Tensor, torch.Tensor],
    tile_size: int = 8,
    opacity_threshold: float = 0.999,
    rendering_mode: Literal["gaussian", "ellipsoid"] = "gaussian",
    confidence: float = 0.95,
) -> torch.Tensor:
    W, H = camera.image_size
    Ht, Wt = H // tile_size, W // tile_size
    device, dtype = splats.mean.device, splats.mean.dtype

    indices, ranges = assignment
    counts = ranges[:, 1] - ranges[:, 0]

    tile_ids = torch.repeat_interleave(torch.arange(Ht * Wt, device=splats.mean.device), counts)

    with timed("Pixel grid"):
        # Get pixel grid for tiles with Gaussians
        tile_pixels, tile_origins = create_pixel_grids_and_origins(tile_size, Ht, Wt, device, dtype)

    with timed("Visible Probability"):
        # Compute the probability of each gaussian being visible at each pixel
        offset = tile_origins[tile_ids, None] + tile_pixels - splats.mean[indices, None, :2]
        distance = mahalanobis_distance_squared(offset, splats.inverse_covariance[indices])
        p = probability(distance, rendering_mode, confidence)
        alpha = (splats.opacity[indices] * p).clamp_max(0.99)

    with timed("Blend"):
        # Get transmittance of each gaussian (at each tile)
        log_survival = torch.log1p(-alpha)
        transmittance = log_survival.cumsum(0) - log_survival
        transmittance = torch.exp(transmittance - transmittance[ranges[tile_ids, 0]])

        # Sum contributions into image (until over opacity threshold)
        weights = transmittance * alpha * ((1 - transmittance) < opacity_threshold)
        contributions = weights[..., None] * splats.color[indices, None]  # M x (T * T) x 3
        image = contributions.new_zeros(Ht * Wt, tile_size * tile_size, 3)
        image = image.index_add(0, tile_ids, contributions)
        return rearrange(
            image,
            "(ht wt) (th tw) c -> (ht th) (wt tw) c",
            ht=Ht,
            wt=Wt,
            th=tile_size,
        )


@torch.no_grad()
def rasterize(
    splats: GaussianSplat,
    camera: Camera,
    #
    tile_size: int = 8,
    opacity_threshold: float = 0.999,
    near: float = 0.1,
    far: float = 100.0,
    num_splats: int | None = None,
    rendering_mode: str = "gaussian",
    confidence: float = 0.95,
) -> np.ndarray:
    projected = project(splats, camera, num_splats)
    culled = cull(projected, near, far)
    tiled = tile(culled, camera, tile_size, rendering_mode, confidence)
    image = render(culled, camera, tiled, tile_size, opacity_threshold, rendering_mode, confidence)
    return (255 * image.clamp(0, 1)).byte().cpu().numpy()
