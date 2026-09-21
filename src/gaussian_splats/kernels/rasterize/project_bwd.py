"""Fused project backward pass for the differentiable projection stage."""

import torch
import triton
import triton.language as tl


@triton.jit
def _project_bwd_kernel(
    mean,
    rotation,
    scale,
    opacity,
    color,
    camera_rotation,
    camera_translation,
    inverse_covariance,
    grad_projected,
    grad_mean,
    grad_rotation,
    grad_scale,
    grad_color_input,
    grad_opacity_input,
    counts,
    fx: tl.constexpr,
    fy: tl.constexpr,
    lim_x: tl.constexpr,
    lim_y: tl.constexpr,
    num_sh: tl.constexpr,
):
    index = tl.program_id(0)
    visible = tl.load(counts + index) != 0
    sh_index = tl.arange(0, 16)
    sh_mask = sh_index < num_sh

    cr00, cr01, cr02 = (
        tl.load(camera_rotation),
        tl.load(camera_rotation + 1),
        tl.load(camera_rotation + 2),
    )
    cr10, cr11, cr12 = (
        tl.load(camera_rotation + 3),
        tl.load(camera_rotation + 4),
        tl.load(camera_rotation + 5),
    )
    cr20, cr21, cr22 = (
        tl.load(camera_rotation + 6),
        tl.load(camera_rotation + 7),
        tl.load(camera_rotation + 8),
    )
    tx, ty, tz = (
        tl.load(camera_translation),
        tl.load(camera_translation + 1),
        tl.load(camera_translation + 2),
    )

    mx, my, mz = (
        tl.load(mean + index * 3),
        tl.load(mean + index * 3 + 1),
        tl.load(mean + index * 3 + 2),
    )
    camera_x = mx * cr00 + my * cr01 + mz * cr02 + tx
    camera_y = mx * cr10 + my * cr11 + mz * cr12 + ty
    camera_z = mx * cr20 + my * cr21 + mz * cr22 + tz

    qw, qx, qy, qz = (
        tl.load(rotation + index * 4),
        tl.load(rotation + index * 4 + 1),
        tl.load(rotation + index * 4 + 2),
        tl.load(rotation + index * 4 + 3),
    )
    qnorm = tl.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    inverse_qnorm = 1.0 / qnorm
    qw, qx, qy, qz = qw * inverse_qnorm, qx * inverse_qnorm, qy * inverse_qnorm, qz * inverse_qnorm
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qw * qz)
    r02 = 2.0 * (qx * qz + qw * qy)
    r10 = 2.0 * (qx * qy + qw * qz)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r12 = 2.0 * (qy * qz - qw * qx)
    r20 = 2.0 * (qx * qz - qw * qy)
    r21 = 2.0 * (qy * qz + qw * qx)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

    inverse_z = 1.0 / camera_z
    normalized_x = camera_x * inverse_z
    normalized_y = camera_y * inverse_z
    x_grad_mul = ((normalized_x >= -lim_x) & (normalized_x <= lim_x)).to(tl.float32)
    y_grad_mul = ((normalized_y >= -lim_y) & (normalized_y <= lim_y)).to(tl.float32)
    covariance_x = tl.minimum(lim_x, tl.maximum(-lim_x, normalized_x)) * camera_z
    covariance_y = tl.minimum(lim_y, tl.maximum(-lim_y, normalized_y)) * camera_z
    j00, j02 = fx * inverse_z, -fx * covariance_x * inverse_z * inverse_z
    j11, j12 = fy * inverse_z, -fy * covariance_y * inverse_z * inverse_z

    # A = J @ camera_rotation @ gaussian_rotation @ diag(exp(scale)).
    jc00 = j00 * cr00 + j02 * cr20
    jc01 = j00 * cr01 + j02 * cr21
    jc02 = j00 * cr02 + j02 * cr22
    jc10 = j11 * cr10 + j12 * cr20
    jc11 = j11 * cr11 + j12 * cr21
    jc12 = j11 * cr12 + j12 * cr22

    sx, sy, sz = (
        tl.exp(tl.load(scale + index * 3)),
        tl.exp(tl.load(scale + index * 3 + 1)),
        tl.exp(tl.load(scale + index * 3 + 2)),
    )
    a00 = (jc00 * r00 + jc01 * r10 + jc02 * r20) * sx
    a01 = (jc00 * r01 + jc01 * r11 + jc02 * r21) * sy
    a02 = (jc00 * r02 + jc01 * r12 + jc02 * r22) * sz
    a10 = (jc10 * r00 + jc11 * r10 + jc12 * r20) * sx
    a11 = (jc10 * r01 + jc11 * r11 + jc12 * r21) * sy
    a12 = (jc10 * r02 + jc11 * r12 + jc12 * r22) * sz

    # Pull the covariance gradient through the 2x2 inverse.
    inv00 = tl.load(inverse_covariance + index * 4)
    inv01 = tl.load(inverse_covariance + index * 4 + 1)
    inv10 = inv01
    inv11 = tl.load(inverse_covariance + index * 4 + 3)
    gi00 = tl.load(grad_projected + index * 9 + 2)
    gi01 = tl.load(grad_projected + index * 9 + 3)
    gi10 = gi01
    gi11 = tl.load(grad_projected + index * 9 + 4)
    h00 = inv00 * gi00 + inv10 * gi10
    h01 = inv00 * gi01 + inv10 * gi11
    h10 = inv01 * gi00 + inv11 * gi10
    h11 = inv01 * gi01 + inv11 * gi11
    gcov00 = -(h00 * inv00 + h01 * inv10)
    gcov01 = -(h00 * inv01 + h01 * inv11)
    gcov10 = -(h10 * inv00 + h11 * inv10)
    gcov11 = -(h10 * inv01 + h11 * inv11)

    sym01 = gcov01 + gcov10
    ga00 = 2.0 * gcov00 * a00 + sym01 * a10
    ga01 = 2.0 * gcov00 * a01 + sym01 * a11
    ga02 = 2.0 * gcov00 * a02 + sym01 * a12
    ga10 = sym01 * a00 + 2.0 * gcov11 * a10
    ga11 = sym01 * a01 + 2.0 * gcov11 * a11
    ga12 = sym01 * a02 + 2.0 * gcov11 * a12

    gb00, gb01, gb02 = ga00 * sx, ga01 * sy, ga02 * sz
    gb10, gb11, gb12 = ga10 * sx, ga11 * sy, ga12 * sz
    grad_sx = ga00 * (jc00 * r00 + jc01 * r10 + jc02 * r20) + ga10 * (
        jc10 * r00 + jc11 * r10 + jc12 * r20
    )
    grad_sy = ga01 * (jc00 * r01 + jc01 * r11 + jc02 * r21) + ga11 * (
        jc10 * r01 + jc11 * r11 + jc12 * r21
    )
    grad_sz = ga02 * (jc00 * r02 + jc01 * r12 + jc02 * r22) + ga12 * (
        jc10 * r02 + jc11 * r12 + jc12 * r22
    )

    # Gradient through Rq in J @ Rc @ Rq.
    gr00 = jc00 * gb00 + jc10 * gb10
    gr01 = jc00 * gb01 + jc10 * gb11
    gr02 = jc00 * gb02 + jc10 * gb12
    gr10 = jc01 * gb00 + jc11 * gb10
    gr11 = jc01 * gb01 + jc11 * gb11
    gr12 = jc01 * gb02 + jc11 * gb12
    gr20 = jc02 * gb00 + jc12 * gb10
    gr21 = jc02 * gb01 + jc12 * gb11
    gr22 = jc02 * gb02 + jc12 * gb12

    gqw = 2.0 * qz * (gr10 - gr01) + 2.0 * qy * (gr02 - gr20) + 2.0 * qx * (gr21 - gr12)
    gqx = (
        2.0 * qy * (gr01 + gr10)
        + 2.0 * qz * (gr02 + gr20)
        + 2.0 * qw * (gr21 - gr12)
        - 4.0 * qx * (gr11 + gr22)
    )
    gqy = (
        2.0 * qx * (gr01 + gr10)
        + 2.0 * qw * (gr02 - gr20)
        + 2.0 * qz * (gr12 + gr21)
        - 4.0 * qy * (gr00 + gr22)
    )
    gqz = (
        2.0 * qw * (gr10 - gr01)
        + 2.0 * qx * (gr02 + gr20)
        + 2.0 * qy * (gr12 + gr21)
        - 4.0 * qz * (gr00 + gr11)
    )
    dot_q = qw * gqw + qx * gqx + qy * gqy + qz * gqz
    gq_scale = 1.0 / qnorm
    gqw = (gqw - qw * dot_q) * gq_scale
    gqx = (gqx - qx * dot_q) * gq_scale
    gqy = (gqy - qy * dot_q) * gq_scale
    gqz = (gqz - qz * dot_q) * gq_scale

    # Gradient through the perspective Jacobian and projected mean.
    gpx = tl.load(grad_projected + index * 9)
    gpy = tl.load(grad_projected + index * 9 + 1)
    gpz = 0.0
    gux = gpx * j00 + gpx * 0.0
    guy = gpy * j11
    guz = gpz + gpx * j02 + gpy * j12
    # gJ = gB @ (camera_rotation @ Rq)^T.
    m00 = cr00 * r00 + cr01 * r10 + cr02 * r20
    m01 = cr00 * r01 + cr01 * r11 + cr02 * r21
    m02 = cr00 * r02 + cr01 * r12 + cr02 * r22
    m10 = cr10 * r00 + cr11 * r10 + cr12 * r20
    m11 = cr10 * r01 + cr11 * r11 + cr12 * r21
    m12 = cr10 * r02 + cr11 * r12 + cr12 * r22
    # gJ = gB @ (camera_rotation @ gaussian_rotation)^T.
    # gJ_{r,c} = sum_j gB_{r,j} (Rc @ Rq)_{c,j}.
    mr0, mr1, mr2 = m00, m01, m02
    nr0, nr1, nr2 = m10, m11, m12
    # Rc @ Rq third row.
    kr0 = cr20 * r00 + cr21 * r10 + cr22 * r20
    kr1 = cr20 * r01 + cr21 * r11 + cr22 * r21
    kr2 = cr20 * r02 + cr21 * r12 + cr22 * r22
    gj_j00 = gb00 * mr0 + gb01 * mr1 + gb02 * mr2
    gj_j02 = gb00 * kr0 + gb01 * kr1 + gb02 * kr2
    gj_j11 = gb10 * nr0 + gb11 * nr1 + gb12 * nr2
    gj_j12 = gb10 * kr0 + gb11 * kr1 + gb12 * kr2

    gux += x_grad_mul * gj_j02 * (-fx * inverse_z * inverse_z)
    guy += y_grad_mul * gj_j12 * (-fy * inverse_z * inverse_z)
    inverse_z3 = inverse_z * inverse_z * inverse_z
    guz += gj_j00 * (-fx * inverse_z * inverse_z) + gj_j02 * 2.0 * fx * covariance_x * inverse_z3
    guz += gj_j11 * (-fy * inverse_z * inverse_z) + gj_j12 * 2.0 * fy * covariance_y * inverse_z3

    gmx = gux * cr00 + guy * cr10 + guz * cr20
    gmy = gux * cr01 + guy * cr11 + guz * cr21
    gmz = gux * cr02 + guy * cr12 + guz * cr22

    # Degree-0..3 SH VJP and direction gradient.
    dx = mx + cr00 * tx + cr10 * ty + cr20 * tz
    dy = my + cr01 * tx + cr11 * ty + cr21 * tz
    dz = mz + cr02 * tx + cr12 * ty + cr22 * tz
    norm = tl.sqrt(dx * dx + dy * dy + dz * dz)
    x, y, z = dx / norm, dy / norm, dz / norm
    xx, yy, zz = x * x, y * y, z * z
    basis = tl.zeros((16,), dtype=tl.float32)
    basis = tl.where(sh_index == 0, 0.28209479177387814, basis)
    basis = tl.where(sh_index == 1, -0.4886025119029199 * y, basis)
    basis = tl.where(sh_index == 2, 0.4886025119029199 * z, basis)
    basis = tl.where(sh_index == 3, -0.4886025119029199 * x, basis)
    basis = tl.where(sh_index == 4, 1.0925484305920792 * x * y, basis)
    basis = tl.where(sh_index == 5, -1.0925484305920792 * y * z, basis)
    basis = tl.where(sh_index == 6, 0.31539156525252005 * (2.0 * zz - xx - yy), basis)
    basis = tl.where(sh_index == 7, -1.0925484305920792 * x * z, basis)
    basis = tl.where(sh_index == 8, 0.5462742152960396 * (xx - yy), basis)
    basis = tl.where(sh_index == 9, -0.5900435899266435 * y * (3.0 * xx - yy), basis)
    basis = tl.where(sh_index == 10, 2.890611442640554 * x * y * z, basis)
    basis = tl.where(sh_index == 11, -0.4570457994644655 * y * (4.0 * zz - xx - yy), basis)
    basis = tl.where(
        sh_index == 12, 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy), basis
    )
    basis = tl.where(sh_index == 13, -0.4570457994644655 * x * (4.0 * zz - xx - yy), basis)
    basis = tl.where(sh_index == 14, 1.445305721320277 * z * (xx - yy), basis)
    basis = tl.where(sh_index == 15, -0.5900435899266435 * x * (xx - 3.0 * yy), basis)

    dbx = tl.zeros((16,), dtype=tl.float32)
    dby = tl.zeros((16,), dtype=tl.float32)
    dbz = tl.zeros((16,), dtype=tl.float32)
    dbx = tl.where(sh_index == 3, -0.4886025119029199, dbx)
    dby = tl.where(sh_index == 1, -0.4886025119029199, dby)
    dbz = tl.where(sh_index == 2, 0.4886025119029199, dbz)
    dbx = tl.where(sh_index == 4, 1.0925484305920792 * y, dbx)
    dby = tl.where(sh_index == 4, 1.0925484305920792 * x, dby)
    dby = tl.where(sh_index == 5, -1.0925484305920792 * z, dby)
    dbz = tl.where(sh_index == 5, -1.0925484305920792 * y, dbz)
    dbx = tl.where(sh_index == 6, -0.6307831301050401 * x, dbx)
    dby = tl.where(sh_index == 6, -0.6307831301050401 * y, dby)
    dbz = tl.where(sh_index == 6, 1.2615662610100802 * z, dbz)
    dbx = tl.where(sh_index == 7, -1.0925484305920792 * z, dbx)
    dbz = tl.where(sh_index == 7, -1.0925484305920792 * x, dbz)
    dbx = tl.where(sh_index == 8, 1.0925484305920792 * x, dbx)
    dby = tl.where(sh_index == 8, -1.0925484305920792 * y, dby)
    dbx = tl.where(sh_index == 9, -3.540261539559861 * x * y, dbx)
    dby = tl.where(sh_index == 9, -1.7701307697799305 * (xx - yy), dby)
    dbx = tl.where(sh_index == 10, 2.890611442640554 * y * z, dbx)
    dby = tl.where(sh_index == 10, 2.890611442640554 * x * z, dby)
    dbz = tl.where(sh_index == 10, 2.890611442640554 * x * y, dbz)
    dbx = tl.where(sh_index == 11, 0.914091599852931 * x * y, dbx)
    dby = tl.where(sh_index == 11, -0.4570457994644655 * (4.0 * zz - xx - 3.0 * yy), dby)
    dbz = tl.where(sh_index == 11, -3.656366399411724 * y * z, dbz)
    dbx = tl.where(sh_index == 12, -2.2390579955366924 * x * z, dbx)
    dby = tl.where(sh_index == 12, -2.2390579955366924 * y * z, dby)
    dbz = tl.where(sh_index == 12, 0.3731763325901154 * (6.0 * zz - 3.0 * xx - 3.0 * yy), dbz)
    dbx = tl.where(sh_index == 13, -0.4570457994644655 * (4.0 * zz - 3.0 * xx - yy), dbx)
    dby = tl.where(sh_index == 13, 0.914091599852931 * x * y, dby)
    dbz = tl.where(sh_index == 13, -3.656366399411724 * x * z, dbz)
    dbx = tl.where(sh_index == 14, 2.890611442640554 * x * z, dbx)
    dby = tl.where(sh_index == 14, -2.890611442640554 * y * z, dby)
    dbz = tl.where(sh_index == 14, 1.445305721320277 * (xx - yy), dbz)
    dbx = tl.where(sh_index == 15, -1.7701307697799305 * (xx - yy), dbx)
    dby = tl.where(sh_index == 15, 3.540261539559861 * x * y, dby)

    gdx, gdy, gdz = 0.0, 0.0, 0.0
    for channel in tl.static_range(3):
        coefficients = tl.load(
            color + index * num_sh * 3 + sh_index * 3 + channel,
            mask=sh_mask,
            other=0.0,
        )
        raw = tl.sum(coefficients * basis) + 0.5
        # The forward SH conversion applies only clamp_min(0), matching the
        # official rasterizer. Values above one are not clamped and retain
        # their gradient.
        gc = tl.load(grad_projected + index * 9 + 5 + channel) * (raw > 0.0)
        tl.store(
            grad_color_input + index * num_sh * 3 + sh_index * 3 + channel,
            tl.where(visible, gc * basis, 0.0),
            mask=sh_mask,
        )
        gdx += gc * tl.sum(coefficients * dbx)
        gdy += gc * tl.sum(coefficients * dby)
        gdz += gc * tl.sum(coefficients * dbz)
    dot_d = x * gdx + y * gdy + z * gdz
    gdx = (gdx - x * dot_d) / norm
    gdy = (gdy - y * dot_d) / norm
    gdz = (gdz - z * dot_d) / norm
    gmx += gdx
    gmy += gdy
    gmz += gdz

    raw_opacity = tl.load(opacity + index)
    projected_opacity = 1.0 / (1.0 + tl.exp(-raw_opacity))
    go = tl.load(grad_projected + index * 9 + 8) * projected_opacity * (1.0 - projected_opacity)

    tl.store(grad_mean + index * 3, tl.where(visible, gmx, 0.0))
    tl.store(grad_mean + index * 3 + 1, tl.where(visible, gmy, 0.0))
    tl.store(grad_mean + index * 3 + 2, tl.where(visible, gmz, 0.0))
    tl.store(grad_rotation + index * 4, tl.where(visible, gqw, 0.0))
    tl.store(grad_rotation + index * 4 + 1, tl.where(visible, gqx, 0.0))
    tl.store(grad_rotation + index * 4 + 2, tl.where(visible, gqy, 0.0))
    tl.store(grad_rotation + index * 4 + 3, tl.where(visible, gqz, 0.0))
    tl.store(grad_scale + index * 3, tl.where(visible, grad_sx * sx, 0.0))
    tl.store(grad_scale + index * 3 + 1, tl.where(visible, grad_sy * sy, 0.0))
    tl.store(grad_scale + index * 3 + 2, tl.where(visible, grad_sz * sz, 0.0))
    tl.store(grad_opacity_input + index, tl.where(visible, go, 0.0))


def project_bwd(
    mean: torch.Tensor,
    rotation: torch.Tensor,
    scale: torch.Tensor,
    opacity: torch.Tensor,
    color: torch.Tensor,
    camera_rotation: torch.Tensor,
    camera_translation: torch.Tensor,
    inverse_covariance: torch.Tensor,
    grad_projected: torch.Tensor,
    counts: torch.Tensor,
    fx: float,
    fy: float,
    image_width: int,
    image_height: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_rotation = torch.empty_like(rotation)
    grad_scale = torch.empty_like(scale)
    grad_color_input = torch.empty_like(color)
    grad_opacity_input = torch.empty_like(opacity)
    grad_mean = torch.empty_like(mean)
    _project_bwd_kernel[(mean.shape[0],)](
        mean,
        rotation,
        scale,
        opacity,
        color,
        camera_rotation,
        camera_translation,
        inverse_covariance,
        grad_projected,
        grad_mean,
        grad_rotation,
        grad_scale,
        grad_color_input,
        grad_opacity_input,
        counts,
        fx=fx,
        fy=fy,
        lim_x=1.3 * image_width / (2.0 * fx),
        lim_y=1.3 * image_height / (2.0 * fy),
        num_sh=color.shape[1],
        num_warps=1,
    )
    return grad_mean, grad_rotation, grad_scale, grad_opacity_input, grad_color_input
