"""Triton tile-compositing kernel."""

import math
import os
from typing import Literal

import torch
import triton
import triton.language as tl


def _autotune(function):
    if os.environ.get("TRITON_INTERPRET") == "1":
        return function
    return triton.autotune(
        configs=[
            triton.Config({}, num_warps=1),
            triton.Config({}, num_warps=2),
            triton.Config({}, num_warps=4),
        ],
        key=["BLOCK_PIXELS", "ELLIPSOID"],
    )(function)


@_autotune
@triton.jit
def _composite_kernel(
    mean,
    inverse_covariance,
    color,
    opacity,
    gaussian_ids,
    ranges,
    image,
    final_transmittance,
    n_contributors,
    opacity_threshold,
    cutoff,
    IMAGE_WIDTH: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    BLOCK_PIXELS: tl.constexpr,
    ELLIPSOID: tl.constexpr,
):
    tile_id = tl.program_id(0)
    pixel_id = tl.arange(0, BLOCK_PIXELS)

    # Get the pixel grid for each tile.
    tiles_w = IMAGE_WIDTH // TILE_SIZE
    pixel_x = (tile_id % tiles_w) * TILE_SIZE + pixel_id % TILE_SIZE
    pixel_y = (tile_id // tiles_w) * TILE_SIZE + pixel_id // TILE_SIZE

    transmittance = tl.full((BLOCK_PIXELS,), 1.0, tl.float32)
    red = tl.zeros((BLOCK_PIXELS,), tl.float32)
    green = tl.zeros((BLOCK_PIXELS,), tl.float32)
    blue = tl.zeros((BLOCK_PIXELS,), tl.float32)
    contributors = tl.zeros((BLOCK_PIXELS,), tl.int32)

    intersection = tl.load(ranges + tile_id * 2)
    end = tl.load(ranges + tile_id * 2 + 1)
    while intersection < end:
        gaussian_id = tl.load(gaussian_ids + intersection)
        mean_offset = gaussian_id * 3
        covariance_offset = gaussian_id * 4
        mean_x = tl.load(mean + mean_offset)
        mean_y = tl.load(mean + mean_offset + 1)
        xx = tl.load(inverse_covariance + covariance_offset)
        xy = tl.load(inverse_covariance + covariance_offset + 1)
        yy = tl.load(inverse_covariance + covariance_offset + 3)

        # Compute the probability of each Gaussian at each pixel.
        dx = pixel_x - mean_x
        dy = pixel_y - mean_y
        distance_squared = xx * dx * dx + 2.0 * xy * dx * dy + yy * dy * dy

        if ELLIPSOID:
            probability = (distance_squared <= cutoff).to(tl.float32)
        else:
            probability = tl.exp(-0.5 * distance_squared)

        # Accumulate transmittance and color contributions.
        alpha = tl.minimum(tl.load(opacity + gaussian_id) * probability, 0.99)
        alpha = tl.where(alpha >= 1.0 / 255.0, alpha, 0.0)

        active = (1.0 - transmittance) < opacity_threshold
        contributors += active.to(tl.int32)
        weight = transmittance * alpha * active

        color_offset = gaussian_id * 3
        color_r = tl.load(color + color_offset)
        color_g = tl.load(color + color_offset + 1)
        color_b = tl.load(color + color_offset + 2)

        red += weight * color_r
        green += weight * color_g
        blue += weight * color_b

        next_transmittance = transmittance * (1.0 - alpha)
        transmittance = tl.where(active, next_transmittance, transmittance)
        finished = tl.min((~active).to(tl.int32), axis=0) == 1
        intersection += tl.where(finished, end - intersection, 1)

    image_offset = (pixel_y * IMAGE_WIDTH + pixel_x) * 3
    tl.store(image + image_offset, red)
    tl.store(image + image_offset + 1, green)
    tl.store(image + image_offset + 2, blue)
    tl.store(final_transmittance + pixel_y * IMAGE_WIDTH + pixel_x, transmittance)
    tl.store(n_contributors + pixel_y * IMAGE_WIDTH + pixel_x, contributors)


def composite(
    mean: torch.Tensor,
    inverse_covariance: torch.Tensor,
    color: torch.Tensor,
    opacity: torch.Tensor,
    gaussian_ids: torch.Tensor,
    ranges: torch.Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    opacity_threshold: float,
    rendering_mode: Literal["gaussian", "ellipsoid"],
    confidence: float,
) -> tuple[
    torch.Tensor,  # image
    torch.Tensor,  # final_transmittance
    torch.Tensor,  # n_contributors
]:
    """Composite sorted per-tile Gaussians into an on-device float image."""
    assert tile_size > 0 and tile_size & (tile_size - 1) == 0
    assert image_width % tile_size == 0 and image_height % tile_size == 0
    tiles_w = image_width // tile_size
    tiles_h = image_height // tile_size

    image = torch.empty((image_height, image_width, 3), device=mean.device, dtype=mean.dtype)
    final_transmittance = mean.new_empty((image_height, image_width))
    n_contributors = mean.new_empty((image_height, image_width), dtype=torch.int32)
    num_tiles = tiles_h * tiles_w
    cutoff = -2.0 * math.log1p(-confidence)

    _composite_kernel[(num_tiles,)](
        mean,
        inverse_covariance,
        color,
        opacity,
        gaussian_ids,
        ranges,
        image,
        final_transmittance,
        n_contributors,
        opacity_threshold,
        cutoff,
        IMAGE_WIDTH=image_width,
        TILE_SIZE=tile_size,
        BLOCK_PIXELS=tile_size * tile_size,
        ELLIPSOID=rendering_mode == "ellipsoid",
    )

    return (
        image,
        final_transmittance,
        n_contributors,
    )
