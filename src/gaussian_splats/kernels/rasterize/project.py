"""Fused Triton projection and tile-count stage."""

from math import log1p, sqrt

import torch
import triton
import triton.language as tl


@triton.jit
def _evaluate_sh(
    color,
    index,
    dx,
    dy,
    dz,
    projected_color,
    num_sh: tl.constexpr,
    sh_block: tl.constexpr,
):
    # TODO: Benchmark direct SH accumulation against this basis-vector implementation.
    xx, yy, zz = dx * dx, dy * dy, dz * dz
    sh_index = tl.arange(0, sh_block)
    basis = tl.full((sh_block,), 0.0, tl.float32)
    basis = tl.where(sh_index == 0, 0.28209479177387814, basis)
    if num_sh > 1:
        basis = tl.where(sh_index == 1, -0.4886025119029199 * dy, basis)
        basis = tl.where(sh_index == 2, 0.4886025119029199 * dz, basis)
        basis = tl.where(sh_index == 3, -0.4886025119029199 * dx, basis)
    if num_sh > 4:
        basis = tl.where(sh_index == 4, 1.0925484305920792 * dx * dy, basis)
        basis = tl.where(sh_index == 5, -1.0925484305920792 * dy * dz, basis)
        basis = tl.where(sh_index == 6, 0.31539156525252005 * (2.0 * zz - xx - yy), basis)
        basis = tl.where(sh_index == 7, -1.0925484305920792 * dx * dz, basis)
        basis = tl.where(sh_index == 8, 0.5462742152960396 * (xx - yy), basis)
    if num_sh > 9:
        basis = tl.where(sh_index == 9, -0.5900435899266435 * dy * (3.0 * xx - yy), basis)
        basis = tl.where(sh_index == 10, 2.890611442640554 * dx * dy * dz, basis)
        basis = tl.where(sh_index == 11, -0.4570457994644658 * dy * (4.0 * zz - xx - yy), basis)
        basis = tl.where(
            sh_index == 12, 0.3731763325901154 * dz * (2.0 * zz - 3.0 * xx - 3.0 * yy), basis
        )
        basis = tl.where(sh_index == 13, -0.4570457994644658 * dx * (4.0 * zz - xx - yy), basis)
        basis = tl.where(sh_index == 14, 1.445305721320277 * dz * (xx - yy), basis)
        basis = tl.where(sh_index == 15, -0.5900435899266435 * dx * (xx - 3.0 * yy), basis)
    if num_sh > 16:
        basis = tl.where(sh_index == 16, 2.5033429417967046 * dx * dy * (xx - yy), basis)
        basis = tl.where(sh_index == 17, -1.7701307697799304 * dy * dz * (3.0 * xx - yy), basis)
        basis = tl.where(sh_index == 18, 0.9461746957575601 * dx * dy * (7.0 * zz - 1.0), basis)
        basis = tl.where(sh_index == 19, -0.6690465435572892 * dy * dz * (7.0 * zz - 3.0), basis)
        basis = tl.where(
            sh_index == 20, 0.10578554691520431 * (zz * (35.0 * zz - 30.0) + 3.0), basis
        )
        basis = tl.where(sh_index == 21, -0.6690465435572892 * dx * dz * (7.0 * zz - 3.0), basis)
        basis = tl.where(sh_index == 22, 0.47308734787878004 * (xx - yy) * (7.0 * zz - 1.0), basis)
        basis = tl.where(sh_index == 23, -1.7701307697799304 * dx * dz * (xx - 3.0 * yy), basis)
        basis = tl.where(
            sh_index == 24,
            0.6258357354491761 * (xx * (xx - 3.0 * yy) - yy * (3.0 * xx - yy)),
            basis,
        )
    sh_mask = sh_index < num_sh
    for channel in tl.static_range(3):
        coefficients = tl.load(
            color + index * num_sh * 3 + sh_index * 3 + channel,
            mask=sh_mask,
            other=0.0,
        )
        value = tl.sum(coefficients * basis) + 0.5
        value = tl.maximum(value, 0.0)
        tl.store(projected_color + index * 3 + channel, value)


@triton.jit
def _project_and_count_kernel(
    mean,
    rotation,
    scale,
    opacity,
    color,
    camera_rotation,
    camera_translation,
    projected_mean,
    inverse_covariance,
    projected_color,
    projected_opacity,
    bounds,
    counts,
    fx: tl.constexpr,
    fy: tl.constexpr,
    cx: tl.constexpr,
    cy: tl.constexpr,
    tiles_w: tl.constexpr,
    tiles_h: tl.constexpr,
    tile_size: tl.constexpr,
    near: tl.constexpr,
    far: tl.constexpr,
    extent: tl.constexpr,
    is_gaussian_render_mode: tl.constexpr,
    num_sh: tl.constexpr,
    sh_block: tl.constexpr,
):
    index = tl.program_id(0)
    mx, my, mz = (
        tl.load(mean + index * 3),
        tl.load(mean + index * 3 + 1),
        tl.load(mean + index * 3 + 2),
    )
    cr00, cr01, cr02, cr10, cr11, cr12, cr20, cr21, cr22 = (
        tl.load(camera_rotation),
        tl.load(camera_rotation + 1),
        tl.load(camera_rotation + 2),
        tl.load(camera_rotation + 3),
        tl.load(camera_rotation + 4),
        tl.load(camera_rotation + 5),
        tl.load(camera_rotation + 6),
        tl.load(camera_rotation + 7),
        tl.load(camera_rotation + 8),
    )
    tx, ty, tz = (
        tl.load(camera_translation),
        tl.load(camera_translation + 1),
        tl.load(camera_translation + 2),
    )

    camera_x, camera_y, camera_z = (
        mx * cr00 + my * cr01 + mz * cr02 + tx,
        mx * cr10 + my * cr11 + mz * cr12 + ty,
        mx * cr20 + my * cr21 + mz * cr22 + tz,
    )
    depth = -camera_z
    safe_z = tl.where(  # clamp abs min to 1e-4
        depth > 0,
        tl.maximum(1e-4, depth),
        tl.minimum(-1e-4, depth),
    )

    # Project Gaussian means and covariances (assumed PinholeCamera).
    pixel_x, pixel_y = (fx * camera_x / safe_z + cx, -fy * camera_y / safe_z + cy)
    tl.store(projected_mean + index * 3, pixel_x)
    tl.store(projected_mean + index * 3 + 1, pixel_y)
    tl.store(projected_mean + index * 3 + 2, safe_z)

    qw, qx, qy, qz = (  # Normalize
        tl.load(rotation + index * 4),
        tl.load(rotation + index * 4 + 1),
        tl.load(rotation + index * 4 + 2),
        tl.load(rotation + index * 4 + 3),
    )
    inverse_norm = tl.rsqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = (
        qw * inverse_norm,
        qx * inverse_norm,
        qy * inverse_norm,
        qz * inverse_norm,
    )

    lim_x = 1.3 * tiles_w * tile_size / (2.0 * fx)
    lim_y = 1.3 * tiles_h * tile_size / (2.0 * fy)
    cov_x = tl.minimum(lim_x, tl.maximum(-lim_x, camera_x / safe_z)) * safe_z
    cov_y = tl.minimum(lim_y, tl.maximum(-lim_y, camera_y / safe_z)) * safe_z
    inverse_z = 1.0 / safe_z
    j00, j02, j11, j12 = (
        fx * inverse_z,
        fx * cov_x * inverse_z * inverse_z,
        -fy * inverse_z,
        -fy * cov_y * inverse_z * inverse_z,
    )
    jc00, jc01, jc02, jc10, jc11, jc12 = (
        j00 * cr00 + j02 * cr20,
        j00 * cr01 + j02 * cr21,
        j00 * cr02 + j02 * cr22,
        j11 * cr10 + j12 * cr20,
        j11 * cr11 + j12 * cr21,
        j11 * cr12 + j12 * cr22,
    )
    sx, sy, sz = (
        tl.exp(tl.load(scale + index * 3)),
        tl.exp(tl.load(scale + index * 3 + 1)),
        tl.exp(tl.load(scale + index * 3 + 2)),
    )

    axis0_x, axis0_y, axis0_z = (
        (1.0 - 2.0 * (qy * qy + qz * qz)) * sx,
        (2.0 * (qx * qy + qw * qz)) * sx,
        (2.0 * (qx * qz - qw * qy)) * sx,
    )
    axis1_x, axis1_y, axis1_z = (
        (2.0 * (qx * qy - qw * qz)) * sy,
        (1.0 - 2.0 * (qx * qx + qz * qz)) * sy,
        (2.0 * (qy * qz + qw * qx)) * sy,
    )
    axis2_x, axis2_y, axis2_z = (
        (2.0 * (qx * qz + qw * qy)) * sz,
        (2.0 * (qy * qz - qw * qx)) * sz,
        (1.0 - 2.0 * (qx * qx + qy * qy)) * sz,
    )

    f00, f01, f02 = (
        jc00 * axis0_x + jc01 * axis0_y + jc02 * axis0_z,
        jc00 * axis1_x + jc01 * axis1_y + jc02 * axis1_z,
        jc00 * axis2_x + jc01 * axis2_y + jc02 * axis2_z,
    )
    f10, f11, f12 = (
        jc10 * axis0_x + jc11 * axis0_y + jc12 * axis0_z,
        jc10 * axis1_x + jc11 * axis1_y + jc12 * axis1_z,
        jc10 * axis2_x + jc11 * axis2_y + jc12 * axis2_z,
    )
    cov00, cov01, cov11 = (
        f00 * f00 + f01 * f01 + f02 * f02 + 0.3,
        f00 * f10 + f01 * f11 + f02 * f12,
        f10 * f10 + f11 * f11 + f12 * f12 + 0.3,
    )
    determinant = cov00 * cov11 - cov01 * cov01
    tl.store(inverse_covariance + index * 4, cov11 / determinant)
    tl.store(inverse_covariance + index * 4 + 1, -cov01 / determinant)
    tl.store(inverse_covariance + index * 4 + 2, -cov01 / determinant)
    tl.store(inverse_covariance + index * 4 + 3, cov00 / determinant)

    # Evaluate spherical harmonics
    dx, dy, dz = (
        mx + cr00 * tx + cr10 * ty + cr20 * tz,
        my + cr01 * tx + cr11 * ty + cr21 * tz,
        mz + cr02 * tx + cr12 * ty + cr22 * tz,
    )
    direction_norm = tl.rsqrt(dx * dx + dy * dy + dz * dz)
    dx, dy, dz = (dx * direction_norm, dy * direction_norm, dz * direction_norm)
    _evaluate_sh(color, index, dx, dy, dz, projected_color, num_sh, sh_block)

    raw_opacity = tl.load(opacity + index)
    _projected_opacity = 1.0 / (1.0 + tl.exp(-raw_opacity))
    tl.store(projected_opacity + index, _projected_opacity)

    radius_extent = (
        2.0 * tl.sqrt(tl.log(_projected_opacity * 255.0)) if is_gaussian_render_mode else extent
    )
    radius_x = radius_extent * tl.sqrt(cov00)
    radius_y = radius_extent * tl.sqrt(cov11)

    # Estimate tiles that Gaussians intersect.
    tile_min_x = tl.clamp(tl.floor((pixel_x - radius_x) / tile_size), 0, tiles_w)
    tile_min_y = tl.clamp(tl.floor((pixel_y - radius_y) / tile_size), 0, tiles_h)
    tile_max_x = tl.clamp(tl.floor((pixel_x + radius_x) / tile_size) + 1, 0, tiles_w)
    tile_max_y = tl.clamp(tl.floor((pixel_y + radius_y) / tile_size) + 1, 0, tiles_h)
    tl.store(bounds + index * 4, tile_min_x)
    tl.store(bounds + index * 4 + 1, tile_max_x)
    tl.store(bounds + index * 4 + 2, tile_min_y)
    tl.store(bounds + index * 4 + 3, tile_max_y)

    # Cull
    count = (tile_max_x - tile_min_x) * (tile_max_y - tile_min_y)
    count = tl.where((safe_z > near) & (safe_z < far), count, 0)
    tl.store(counts + index, count)


def project_and_count(
    mean: torch.Tensor,
    rotation: torch.Tensor,
    scale: torch.Tensor,
    opacity: torch.Tensor,
    color: torch.Tensor,
    camera_rotation: torch.Tensor,
    camera_translation: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_width: int,
    image_height: int,
    tile_size: int,
    near: float,
    far: float,
    rendering_mode: str,
    confidence: float,
) -> tuple[
    torch.Tensor,  # projected_mean
    torch.Tensor,  # inverse_covariance
    torch.Tensor,  # projected_color
    torch.Tensor,  # projected_opacity
    torch.Tensor,  # bounds
    torch.Tensor,  # counts
]:
    """Project Gaussians and count their intersected tiles without compacting."""
    assert tile_size > 0 and tile_size & (tile_size - 1) == 0
    assert image_width % tile_size == 0 and image_height % tile_size == 0

    if color.shape[1] > 25:
        raise NotImplementedError(f"SH degrees above 4 are not implemented. Got {color.shape[1]}.")

    N = len(mean)
    projected_mean = mean.new_empty((N, 3))
    inverse_covariance = mean.new_empty((N, 2, 2))
    projected_color = mean.new_empty((N, 3))
    projected_opacity = mean.new_empty((N, 1))
    bounds = torch.empty((N, 2, 2), dtype=torch.int32, device=mean.device)
    counts = torch.empty(N, dtype=torch.int32, device=mean.device)

    if N == 0:
        return (
            projected_mean,
            inverse_covariance,
            projected_color,
            projected_opacity,
            bounds,
            counts,
        )

    extent = sqrt(-2 * log1p(-confidence)) if rendering_mode == "ellipsoid" else 3.0
    _project_and_count_kernel[(N,)](
        mean,
        rotation,
        scale,
        opacity,
        color,
        camera_rotation,
        camera_translation,
        projected_mean,
        inverse_covariance,
        projected_color,
        projected_opacity,
        bounds,
        counts,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        tiles_w=image_width // tile_size,
        tiles_h=image_height // tile_size,
        tile_size=tile_size,
        near=near,
        far=far,
        extent=extent,
        is_gaussian_render_mode=rendering_mode == "gaussian",
        num_sh=color.shape[1],
        sh_block=triton.next_power_of_2(color.shape[1]),
        num_warps=1,
    )
    return (
        projected_mean,
        inverse_covariance,
        projected_color,
        projected_opacity,
        bounds,
        counts,
    )
