"""Pure-PyTorch reference stages used to validate compiled rasterizers."""

from __future__ import annotations

import warnings
from math import log1p, sqrt

import torch
import torch.nn.functional as F
from einops import rearrange

from ...splats.utils import (
    evaluate_sh,
    mahalanobis_distance_squared,
    probability,
    quaternion_to_rotation_matrix,
)


def project_gaussians(
    mean,
    rotation,
    scale,
    opacity,
    color,
    camera_rotation,
    camera_translation,
    fx,
    fy,
    cx,
    cy,
    image_width,
    image_height,
):
    # Project Gaussian means and covariances.
    mean_world = mean
    mean = mean @ camera_rotation.T + camera_translation
    x, y, z = mean.unbind(dim=-1)
    z = torch.where(z.abs() >= 1e-4, z, torch.sign(z) * 1e-4)
    intrinsics = mean.new_tensor(((fx, 0, cx), (0, fy, cy), (0, 0, 1)))
    mean = mean @ intrinsics.T
    mean = mean[:, :2] / mean[:, 2:]
    mean = torch.cat((mean, z[:, None]), dim=-1)

    tan_half_fov_x = image_width / (2 * fx)
    tan_half_fov_y = image_height / (2 * fy)
    zero = torch.zeros_like(z)
    jacobian = torch.stack(
        (
            fx / z,
            zero,
            -fx * (x / z).clamp(-1.3 * tan_half_fov_x, 1.3 * tan_half_fov_x) / z,
            zero,
            fy / z,
            -fy * (y / z).clamp(-1.3 * tan_half_fov_y, 1.3 * tan_half_fov_y) / z,
        ),
        dim=-1,
    ).view(-1, 2, 3)
    covariance_factor = (
        jacobian
        @ camera_rotation
        @ quaternion_to_rotation_matrix(F.normalize(rotation, dim=-1))
        * scale.exp()[:, None]
    )
    covariance = covariance_factor @ covariance_factor.mT
    covariance.diagonal(dim1=-2, dim2=-1).add_(0.3)

    # Evaluate spherical harmonics and compute opacity.
    camera_position = -(camera_rotation.T @ camera_translation)
    direction = F.normalize(mean_world - camera_position, dim=-1)
    color = evaluate_sh(color, direction)
    opacity = opacity.sigmoid()
    inverse_covariance = torch.linalg.inv(covariance)
    return mean, covariance, inverse_covariance, color, opacity


def project_and_count(
    mean,
    rotation,
    scale,
    opacity,
    color,
    camera_rotation,
    camera_translation,
    fx,
    fy,
    cx,
    cy,
    image_width,
    image_height,
    tile_size,
    near,
    far,
    rendering_mode,
    confidence,
):
    assert tile_size > 0 and tile_size & (tile_size - 1) == 0
    assert image_width % tile_size == 0 and image_height % tile_size == 0
    projected_mean, covariance, inverse_covariance, color, opacity = project_gaussians(
        mean,
        rotation,
        scale,
        opacity,
        color,
        camera_rotation,
        camera_translation,
        fx,
        fy,
        cx,
        cy,
        image_width,
        image_height,
    )
    W, H = image_width, image_height
    Ht, Wt = H // tile_size, W // tile_size

    if rendering_mode == "gaussian":
        extent = (2 * torch.log(opacity.squeeze(-1) * 255).sqrt().clamp_min(0))[:, None]
    elif rendering_mode == "ellipsoid":
        extent = sqrt(-2 * log1p(-confidence))
    else:
        raise ValueError(f"Invalid rendering mode: {rendering_mode}")

    radius = extent * covariance.diagonal(dim1=-2, dim2=-1).sqrt()

    # Estimate tiles that Gaussians intersect
    bounds = torch.stack((projected_mean[:, :2] - radius, projected_mean[:, :2] + radius), dim=-1)
    bounds = (bounds / tile_size).floor().int()
    bounds[:, :, 1] += 1  # N x 2 x 2 (N, x/y, min/max)

    limits = bounds.new_tensor((Wt, Ht))[:, None]
    bounds.clamp_(torch.zeros_like(limits), limits)

    tile_min, tile_max = bounds[:, :, 0], bounds[:, :, 1]
    size = tile_max - tile_min
    counts = size.prod(dim=-1)

    counts *= (projected_mean[:, 2] > near) & (projected_mean[:, 2] < far)
    return projected_mean, inverse_covariance, color, opacity, bounds, counts


def emit_intersections(bounds, depths, counts, offsets, tiles_w):
    tile_min, tile_max = bounds[:, :, 0], bounds[:, :, 1]
    size = tile_max - tile_min
    # Assign Gaussian IDs to tiles
    gaussian_ids = torch.repeat_interleave(
        torch.arange(len(depths), device=depths.device, dtype=torch.int32), counts
    )
    local_id = torch.arange(counts.sum(), device=depths.device)
    local_id -= torch.repeat_interleave(offsets, counts)
    origins = tile_min[gaussian_ids]
    widths = size[gaussian_ids, 0]
    # Get corresponding tile IDs
    tile_ids = (origins[:, 1] + local_id // widths) * tiles_w + origins[:, 0] + local_id % widths
    depths = depths[gaussian_ids]
    if depths.dtype != torch.float32:
        warnings.warn(
            f"Converting intersection depths from {depths.dtype} to float32 for key packing.",
            RuntimeWarning,
            stacklevel=2,
        )
        depths = depths.float()
    depth_bits = depths.contiguous().view(torch.int32).to(torch.int64)
    keys = (tile_ids.to(torch.int64) << 32) | (depth_bits & 0xFFFFFFFF)
    # Order by tile ID x depth
    keys, order = keys.sort()
    gaussian_ids = gaussian_ids[order]
    return keys, gaussian_ids


def sort_intersections(keys, gaussian_ids, num_tiles):
    # Order by tile ID x depth
    keys, order = keys.sort()
    tile_ids = keys >> 32
    # Count Gaussians per tile
    counts = torch.bincount(tile_ids, minlength=num_tiles).to(torch.int32)
    ends = counts.cumsum(0).to(torch.int32)
    return gaussian_ids[order], torch.stack((ends - counts, ends), -1)


def composite(
    mean,
    inverse_covariance,
    color,
    opacity,
    indices,
    ranges,
    image_width,
    image_height,
    tile_size,
    opacity_threshold,
    rendering_mode,
    confidence,
):
    (
        mean,
        inverse_covariance,
        color,
        opacity,
    ) = (mean, inverse_covariance, color, opacity)

    W, H = image_width, image_height
    assert tile_size > 0 and tile_size & (tile_size - 1) == 0
    assert W % tile_size == 0 and H % tile_size == 0
    Ht, Wt = H // tile_size, W // tile_size
    counts = ranges[:, 1] - ranges[:, 0]

    # Get the pixel grid for each tile.
    tile_ids = torch.repeat_interleave(torch.arange(Ht * Wt, device=mean.device), counts)
    y, x = torch.meshgrid(
        torch.arange(tile_size, device=mean.device, dtype=mean.dtype),
        torch.arange(tile_size, device=mean.device, dtype=mean.dtype),
        indexing="ij",
    )
    tile_y, tile_x = torch.meshgrid(
        torch.arange(Ht, device=mean.device, dtype=mean.dtype),
        torch.arange(Wt, device=mean.device, dtype=mean.dtype),
        indexing="ij",
    )
    tile_pixels = rearrange(torch.stack((x, y), -1), "h w c -> (h w) c")
    tile_origins = tile_size * rearrange(torch.stack((tile_x, tile_y), -1), "h w c -> (h w) c")

    # Compute the probability of each Gaussian at each pixel.
    offset = tile_origins[tile_ids, None] + tile_pixels - mean[indices, None, :2]
    distance = mahalanobis_distance_squared(offset, inverse_covariance[indices])
    alpha = (opacity[indices] * probability(distance, rendering_mode, confidence)).clamp_max(0.99)
    alpha = torch.where(alpha >= 1.0 / 255.0, alpha, 0.0)

    # Accumulate transmittance and color contributions.
    log_survival = torch.log1p(-alpha)
    transmittance = log_survival.cumsum(0) - log_survival
    transmittance = torch.exp(transmittance - transmittance[ranges[tile_ids, 0]])

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
