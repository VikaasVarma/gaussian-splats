"""Triton composite backward pass for the rasterizer."""

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
        key=["BLOCK_PIXELS"],
    )(function)


@triton.jit
def _combine_reverse(a_t, a_r, a_g, a_b, b_t, b_r, b_g, b_b):
    """Compose reverse-ordered suffix elements.

    ``a`` is the later element and ``b`` is the current element, so the
    accumulated color is ``b + b_t * a``.
    """
    return (
        a_t * b_t,
        b_r + b_t * a_r,
        b_g + b_t * a_g,
        b_b + b_t * a_b,
    )


@triton.jit
def _write_gradient(pointer, value, mask, atomic: tl.constexpr):
    if atomic:
        tl.atomic_add(pointer, value, mask=mask)
    else:
        tl.store(pointer, value, mask=mask)


@_autotune
@triton.jit
def _composite_bwd_kernel(
    projected_mean,
    inverse_covariance,
    projected_color,
    projected_opacity,
    gaussian_ids,
    ranges,
    grad_image,
    final_transmittance,
    n_contributors,
    grad_occ,
    grad_abs_occ,
    opacity_threshold,
    IMAGE_WIDTH: tl.constexpr,
    IMAGE_HEIGHT: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    BLOCK_PIXELS: tl.constexpr,
    CHUNK: tl.constexpr,
    ATOMIC: tl.constexpr,
):
    tile_id = tl.program_id(0)
    pixel_id = tl.arange(0, BLOCK_PIXELS)
    tiles_w = IMAGE_WIDTH // TILE_SIZE
    pixel_x = (tile_id % tiles_w) * TILE_SIZE + pixel_id % TILE_SIZE
    pixel_y = (tile_id // tiles_w) * TILE_SIZE + pixel_id // TILE_SIZE
    image_offset = (pixel_y * IMAGE_WIDTH + pixel_x) * 3
    grad_r = tl.load(grad_image + image_offset)
    grad_g = tl.load(grad_image + image_offset + 1)
    grad_b = tl.load(grad_image + image_offset + 2)

    start = tl.load(ranges + tile_id * 2)
    end = tl.load(ranges + tile_id * 2 + 1)
    contributor_count = tl.load(n_contributors + pixel_y * IMAGE_WIDTH + pixel_x)
    max_contributors = tl.max(contributor_count, axis=0)
    end = tl.minimum(end, start + max_contributors)
    final_t = tl.load(
        final_transmittance + pixel_y * IMAGE_WIDTH + pixel_x,
    )
    carry_t = tl.full((BLOCK_PIXELS,), 1.0, tl.float32)
    carry_r = tl.zeros((BLOCK_PIXELS,), tl.float32)
    carry_g = tl.zeros((BLOCK_PIXELS,), tl.float32)
    carry_b = tl.zeros((BLOCK_PIXELS,), tl.float32)
    chunk_id = tl.arange(0, CHUNK)
    offset = 0
    while offset < end - start:
        intersection = end - 1 - offset - chunk_id
        valid = intersection >= start
        gaussian_id = tl.load(gaussian_ids + intersection, mask=valid, other=0)
        target = gaussian_id if ATOMIC else intersection
        mean_offset = gaussian_id[:, None] * 3
        covariance_offset = gaussian_id[:, None] * 4
        color_offset = gaussian_id[:, None] * 3
        dx = pixel_x[None, :] - tl.load(
            projected_mean + mean_offset, mask=valid[:, None], other=0.0
        )
        dy = pixel_y[None, :] - tl.load(
            projected_mean + mean_offset + 1, mask=valid[:, None], other=0.0
        )
        xx = tl.load(inverse_covariance + covariance_offset, mask=valid[:, None], other=0.0)
        xy = tl.load(inverse_covariance + covariance_offset + 1, mask=valid[:, None], other=0.0)
        yy = tl.load(inverse_covariance + covariance_offset + 3, mask=valid[:, None], other=0.0)
        distance = xx * dx * dx + 2.0 * xy * dx * dy + yy * dy * dy
        probability = tl.exp(-0.5 * distance)
        projected_opacity = tl.load(
            projected_opacity + gaussian_id[:, None], mask=valid[:, None], other=0.0
        )
        alpha_raw = projected_opacity * probability
        alpha = tl.minimum(alpha_raw, 0.99)
        position = intersection - start
        active = (
            valid[:, None]
            & (position[:, None] < contributor_count[None, :])
            & (alpha >= 1.0 / 255.0)
        )
        effective_alpha = tl.where(active, alpha, 0.0)
        local_t = 1.0 - effective_alpha
        color_r = tl.load(projected_color + color_offset, mask=valid[:, None], other=0.0)
        color_g = tl.load(projected_color + color_offset + 1, mask=valid[:, None], other=0.0)
        color_b = tl.load(projected_color + color_offset + 2, mask=valid[:, None], other=0.0)
        local_r = effective_alpha * color_r
        local_g = effective_alpha * color_g
        local_b = effective_alpha * color_b
        scan_t, scan_r, scan_g, scan_b = tl.associative_scan(
            (local_t, local_r, local_g, local_b), axis=0, combine_fn=_combine_reverse
        )
        previous_t = scan_t / local_t
        previous_r = (scan_r - local_r) / local_t
        previous_g = (scan_g - local_g) / local_t
        previous_b = (scan_b - local_b) / local_t
        suffix_t = previous_t * carry_t[None, :]
        suffix_r = previous_r + previous_t * carry_r[None, :]
        suffix_g = previous_g + previous_t * carry_g[None, :]
        suffix_b = previous_b + previous_t * carry_b[None, :]
        denominator = tl.maximum(suffix_t * (1.0 - alpha), 0.0001)
        forward = final_t[None, :] / denominator
        active = active & (alpha >= 1.0 / 255.0) & (1.0 - forward < opacity_threshold)
        weight_grad = grad_r[None, :] * (color_r - suffix_r)
        weight_grad += grad_g[None, :] * (color_g - suffix_g)
        weight_grad += grad_b[None, :] * (color_b - suffix_b)
        grad_alpha = tl.where(active, forward * weight_grad, 0.0)
        grad_alpha = tl.where(alpha_raw < 0.99, grad_alpha, 0.0)
        probability_grad = grad_alpha * projected_opacity
        grad_opacity_value = grad_alpha * probability
        grad_color_value = forward * effective_alpha
        _write_gradient(
            grad_occ + target * 9 + 8,
            tl.sum(grad_opacity_value, axis=1),
            valid,
            ATOMIC,
        )
        _write_gradient(
            grad_occ + target * 9 + 5,
            tl.sum(grad_color_value * grad_r[None, :], axis=1),
            valid,
            ATOMIC,
        )
        _write_gradient(
            grad_occ + target * 9 + 6,
            tl.sum(grad_color_value * grad_g[None, :], axis=1),
            valid,
            ATOMIC,
        )
        _write_gradient(
            grad_occ + target * 9 + 7,
            tl.sum(grad_color_value * grad_b[None, :], axis=1),
            valid,
            ATOMIC,
        )
        grad_distance = -0.5 * probability * probability_grad
        mean_gradient_x = tl.sum(-2.0 * grad_distance * (xx * dx + xy * dy), axis=1)
        mean_gradient_y = tl.sum(-2.0 * grad_distance * (xy * dx + yy * dy), axis=1)
        absolute_mean_gradient_x = tl.sum(
            tl.abs(-2.0 * grad_distance * (xx * dx + xy * dy)), axis=1
        )
        absolute_mean_gradient_y = tl.sum(
            tl.abs(-2.0 * grad_distance * (xy * dx + yy * dy)), axis=1
        )
        covariance_gradient_xx = tl.sum(grad_distance * dx * dx, axis=1)
        covariance_gradient_xy = tl.sum(grad_distance * dx * dy, axis=1)
        covariance_gradient_yy = tl.sum(grad_distance * dy * dy, axis=1)
        _write_gradient(grad_occ + target * 9, mean_gradient_x, valid, ATOMIC)
        _write_gradient(grad_occ + target * 9 + 1, mean_gradient_y, valid, ATOMIC)
        _write_gradient(grad_occ + target * 9 + 2, covariance_gradient_xx, valid, ATOMIC)
        _write_gradient(grad_occ + target * 9 + 3, covariance_gradient_xy, valid, ATOMIC)
        _write_gradient(grad_occ + target * 9 + 4, covariance_gradient_yy, valid, ATOMIC)
        _write_gradient(grad_abs_occ + target * 2, absolute_mean_gradient_x, valid, ATOMIC)
        _write_gradient(grad_abs_occ + target * 2 + 1, absolute_mean_gradient_y, valid, ATOMIC)
        last_t = tl.sum(tl.where(chunk_id[:, None] == CHUNK - 1, scan_t, 0.0), axis=0)
        last_r = tl.sum(tl.where(chunk_id[:, None] == CHUNK - 1, scan_r, 0.0), axis=0)
        last_g = tl.sum(tl.where(chunk_id[:, None] == CHUNK - 1, scan_g, 0.0), axis=0)
        last_b = tl.sum(tl.where(chunk_id[:, None] == CHUNK - 1, scan_b, 0.0), axis=0)
        carry_r = last_r + last_t * carry_r
        carry_g = last_g + last_t * carry_g
        carry_b = last_b + last_t * carry_b
        carry_t *= last_t
        offset += CHUNK


def composite_bwd(
    projected_mean: torch.Tensor,
    inverse_covariance: torch.Tensor,
    projected_color: torch.Tensor,
    projected_opacity: torch.Tensor,
    gaussian_ids: torch.Tensor,
    ranges: torch.Tensor,
    grad_image: torch.Tensor,
    final_transmittance: torch.Tensor,
    n_contributors: torch.Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    opacity_threshold: float,
    rendering_mode: Literal["gaussian", "ellipsoid"],
    atomic: bool = False,
    return_absolute: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    assert rendering_mode == "gaussian", (
        "Backward gradients are only supported for Gaussian rendering."
    )
    assert tile_size > 0 and tile_size & (tile_size - 1) == 0
    assert image_width % tile_size == 0 and image_height % tile_size == 0

    count = projected_mean.shape[0]
    block_pixels = tile_size * tile_size
    output_count = gaussian_ids.numel()
    # Tiles can terminate before their final emitted intersections, so the
    # reverse kernel intentionally leaves some rows untouched. Clear those
    # rows before accumulating the written prefix.
    grad_occ = torch.zeros(
        (count if atomic else output_count, 9),
        device=projected_mean.device,
        dtype=projected_mean.dtype,
    )
    grad_abs_occ = torch.zeros(
        (count if atomic else output_count, 2),
        device=projected_mean.device,
        dtype=projected_mean.dtype,
    )
    grad = torch.zeros((count, 9), device=projected_mean.device, dtype=projected_mean.dtype)
    _composite_bwd_kernel[((image_height // tile_size) * (image_width // tile_size),)](
        projected_mean,
        inverse_covariance,
        projected_color,
        projected_opacity,
        gaussian_ids,
        ranges,
        grad_image,
        final_transmittance,
        n_contributors,
        grad_occ,
        grad_abs_occ,
        opacity_threshold,
        IMAGE_WIDTH=image_width,
        IMAGE_HEIGHT=image_height,
        TILE_SIZE=tile_size,
        BLOCK_PIXELS=block_pixels,
        CHUNK=int(os.environ.get("GAUSSIAN_SPLATS_BACKWARD_CHUNK", "8")),
        ATOMIC=atomic,
    )
    if not atomic:
        # CUDA index_add accepts int32 indices; keep the compact IDs emitted
        # by the tile pass instead of allocating an int64 conversion.
        grad.index_add_(0, gaussian_ids, grad_occ)
        absolute = torch.zeros((count, 2), device=grad_abs_occ.device, dtype=grad_abs_occ.dtype)
        absolute.index_add_(0, gaussian_ids, grad_abs_occ)
    else:
        grad = grad_occ
        absolute = grad_abs_occ
    return (grad, absolute) if return_absolute else grad
