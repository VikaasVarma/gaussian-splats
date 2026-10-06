import math

import torch

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)
SH_C4 = (
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
)


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert a quaternion to a rotation matrix. Assumes normalized quaternions."""
    w, x, y, z = quaternion.unbind(dim=-1)

    return torch.stack(
        (
            1 - 2 * (y.square() + z.square()),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x.square() + z.square()),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x.square() + y.square()),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / SH_C0


def spherical_harmonics(
    directions: torch.Tensor,  # N x 3
    num_coefficients: int,
) -> torch.Tensor:
    """Evaluate real spherical-harmonic basis functions through degree 6."""
    degree = int(num_coefficients**0.5) - 1
    if (degree + 1) ** 2 != num_coefficients or degree > 6:
        raise ValueError(f"Unsupported spherical-harmonic coefficient count: {num_coefficients}")

    x, y, z = directions.unbind(dim=-1)
    xx, yy, zz = x.square(), y.square(), z.square()
    xy, yz, xz = x * y, y * z, x * z

    basis = directions.new_empty((len(directions), num_coefficients))
    basis[:, 0] = SH_C0

    if num_coefficients > 1:
        basis[:, 1:4] = torch.stack((-SH_C1 * y, SH_C1 * z, -SH_C1 * x), dim=-1)
    if num_coefficients > 4:
        basis[:, 4:9] = torch.stack(
            (
                SH_C2[0] * xy,
                SH_C2[1] * yz,
                SH_C2[2] * (2 * zz - xx - yy),
                SH_C2[3] * xz,
                SH_C2[4] * (xx - yy),
            ),
            dim=-1,
        )
    if num_coefficients > 9:
        basis[:, 9:16] = torch.stack(
            (
                SH_C3[0] * y * (3 * xx - yy),
                SH_C3[1] * xy * z,
                SH_C3[2] * y * (4 * zz - xx - yy),
                SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy),
                SH_C3[4] * x * (4 * zz - xx - yy),
                SH_C3[5] * z * (xx - yy),
                SH_C3[6] * x * (xx - 3 * yy),
            ),
            dim=-1,
        )
    if num_coefficients > 16:
        basis[:, 16:25] = torch.stack(
            (
                SH_C4[0] * xy * (xx - yy),
                SH_C4[1] * yz * (3 * xx - yy),
                SH_C4[2] * xy * (7 * zz - 1),
                SH_C4[3] * yz * (7 * zz - 3),
                SH_C4[4] * (zz * (35 * zz - 30) + 3),
                SH_C4[5] * xz * (7 * zz - 3),
                SH_C4[6] * (xx - yy) * (7 * zz - 1),
                SH_C4[7] * xz * (xx - 3 * yy),
                SH_C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)),
            ),
            dim=-1,
        )
    if num_coefficients > 25:
        basis[:, 25:36] = torch.stack(
            (
                z * (63 * zz.square() - 70 * zz + 15),
                x * (63 * zz.square() - 42 * zz + 3),
                y * (63 * zz.square() - 42 * zz + 3),
                (xx - yy) * (9 * zz - 1),
                2 * xy * (9 * zz - 1),
                z * x * (xx - 3 * yy),
                z * y * (3 * xx - yy),
                xx.square() - 6 * xx * yy + yy.square(),
                4 * xy * (xx - yy),
                x * (xx.square() - 10 * xx * yy + 5 * yy.square()),
                y * (5 * xx.square() - 10 * xx * yy + yy.square()),
            ),
            dim=-1,
        )
    if num_coefficients > 36:
        basis[:, 36:49] = torch.stack(
            (
                231 * zz.pow(3) - 315 * zz.square() + 105 * zz - 5,
                x * z * (231 * zz.square() - 210 * zz + 35),
                y * z * (231 * zz.square() - 210 * zz + 35),
                (xx - yy) * (33 * zz.square() - 18 * zz + 1),
                2 * xy * (33 * zz.square() - 18 * zz + 1),
                x * (xx - 3 * yy) * (11 * zz - 1),
                y * (3 * xx - yy) * (11 * zz - 1),
                (xx.square() - 6 * xx * yy + yy.square()) * (11 * zz - 1),
                4 * xy * (xx - yy) * (11 * zz - 1),
                z * x * (xx.square() - 10 * xx * yy + 5 * yy.square()),
                z * y * (5 * xx.square() - 10 * xx * yy + yy.square()),
                x.pow(6) - 15 * x.pow(4) * yy + 15 * xx * yy.square() - yy.pow(3),
                2 * xy * (3 * x.pow(4) - 10 * xx * yy + 3 * yy.square()),
            ),
            dim=-1,
        )
    return basis


def evaluate_sh(
    coefficients: torch.Tensor,  # N x (degree + 1)^2 x 3 (RGB)
    directions: torch.Tensor,  # N x 3
) -> torch.Tensor:
    """Evaluate spherical-harmonic RGB coefficients."""
    basis = spherical_harmonics(directions, coefficients.shape[1])
    return ((basis.unsqueeze(1) @ coefficients).squeeze(1) + 0.5).clamp_min(0)


@torch.compile(fullgraph=True, dynamic=True)
def mahalanobis_distance_squared(
    offset: torch.Tensor,  # M x (tile_size * tile_size) x 2
    inverse_covariance: torch.Tensor,  # M x 2 x 2
) -> torch.Tensor:
    x, y = offset.unbind(dim=-1)
    xx = inverse_covariance[..., None, 0, 0]
    xy = inverse_covariance[..., None, 0, 1]
    yy = inverse_covariance[..., None, 1, 1]
    return xx * x.square() + 2 * xy * x * y + yy * y.square()


@torch.compile(fullgraph=True, dynamic=True)
def probability(
    distance_squared: torch.Tensor,
    rendering_mode: str = "gaussian",
    confidence: float = 0.95,
) -> torch.Tensor:
    if rendering_mode == "ellipsoid":
        cutoff = -2 * math.log1p(-confidence)
        return (distance_squared <= cutoff).to(distance_squared.dtype)
    return torch.exp(-0.5 * distance_squared)
