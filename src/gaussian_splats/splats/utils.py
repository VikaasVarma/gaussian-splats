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

SH_C5 = (
    -0.6563820568401701,
    2.0756623148810416,
    -0.4892382994352504,
    2.396768392486662,
    -0.45294665119569694,
    0.1169503224534236,
    -0.45294665119569694,
    2.396768392486662,
    -0.4892382994352504,
    2.0756623148810416,
    -0.6563820568401701,
)
SH_C6 = (
    0.6831841051919143,
    -2.366619162231752,
    0.5045649007287242,
    -0.9212052595149236,
    0.46060262975746175,
    -0.5826213625187314,
    0.06356920226762842,
    -0.5826213625187314,
    0.46060262975746175,
    -0.9212052595149236,
    0.5045649007287242,
    -2.366619162231752,
    0.6831841051919143,
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
    """Evaluate real spherical-harmonic basis functions through degree 8."""
    degree = int(num_coefficients**0.5) - 1
    if (degree + 1) ** 2 != num_coefficients or degree > 8:
        raise ValueError(f"Unsupported spherical-harmonic coefficient count: {num_coefficients}")

    x, y, z = directions.unbind(dim=-1)
    if degree > 6:
        # Associated Legendre recurrence in Cartesian form, including at the poles.
        basis = directions.new_empty((len(directions), num_coefficients))
        real, imaginary = torch.ones_like(z), torch.zeros_like(z)
        for order in range(degree + 1):
            if order:
                factor = -(2 * order - 1)
                real, imaginary = (
                    factor * (x * real - y * imaginary),
                    factor * (y * real + x * imaginary),
                )
            previous = None
            current = torch.stack((real, imaginary), dim=-1)
            for band in range(order, degree + 1):
                if band == order:
                    value = current
                elif band == order + 1:
                    value = (2 * order + 1) * z[:, None] * current
                else:
                    value = (
                        (2 * band - 1) * z[:, None] * current - (band + order - 1) * previous
                    ) / (band - order)
                if band > order:
                    previous, current = current, value
                scale = math.sqrt(
                    (2 * band + 1)
                    / (4 * math.pi)
                    * math.factorial(band - order)
                    / math.factorial(band + order)
                )
                if order:
                    basis[:, band * band + band - order] = math.sqrt(2) * scale * value[:, 1]
                    basis[:, band * band + band + order] = math.sqrt(2) * scale * value[:, 0]
                else:
                    basis[:, band * band + band] = scale * value[:, 0]
        return basis

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
    # Normalized real SH, ordered by m = -l, ..., l, as in the lower bands.
    # Derived from https://dlmf.nist.gov/14.30.E1 and 14.7.E8.
    if num_coefficients > 25:
        basis[:, 25:36] = torch.stack(
            (
                SH_C5[0] * y * (5 * xx.square() - 10 * xx * yy + yy.square()),
                SH_C5[1] * 4 * xy * z * (xx - yy),
                SH_C5[2] * y * (3 * xx - yy) * (9 * zz - 1),
                SH_C5[3] * 2 * xy * z * (3 * zz - 1),
                SH_C5[4] * y * (21 * zz.square() - 14 * zz + 1),
                SH_C5[5] * z * (63 * zz.square() - 70 * zz + 15),
                SH_C5[6] * x * (21 * zz.square() - 14 * zz + 1),
                SH_C5[7] * (xx - yy) * z * (3 * zz - 1),
                SH_C5[8] * x * (xx - 3 * yy) * (9 * zz - 1),
                SH_C5[9] * z * (xx.square() - 6 * xx * yy + yy.square()),
                SH_C5[10] * x * (xx.square() - 10 * xx * yy + 5 * yy.square()),
            ),
            dim=-1,
        )
    if num_coefficients > 36:
        basis[:, 36:49] = torch.stack(
            (
                SH_C6[0] * 2 * xy * (3 * xx.square() - 10 * xx * yy + 3 * yy.square()),
                SH_C6[1] * z * y * (5 * xx.square() - 10 * xx * yy + yy.square()),
                SH_C6[2] * 4 * xy * (xx - yy) * (11 * zz - 1),
                SH_C6[3] * z * y * (3 * xx - yy) * (11 * zz - 3),
                SH_C6[4] * 2 * xy * (33 * zz.square() - 18 * zz + 1),
                SH_C6[5] * y * z * (33 * zz.square() - 30 * zz + 5),
                SH_C6[6] * (231 * zz.pow(3) - 315 * zz.square() + 105 * zz - 5),
                SH_C6[7] * x * z * (33 * zz.square() - 30 * zz + 5),
                SH_C6[8] * (xx - yy) * (33 * zz.square() - 18 * zz + 1),
                SH_C6[9] * z * x * (xx - 3 * yy) * (11 * zz - 3),
                SH_C6[10] * (xx.square() - 6 * xx * yy + yy.square()) * (11 * zz - 1),
                SH_C6[11] * z * x * (xx.square() - 10 * xx * yy + 5 * yy.square()),
                SH_C6[12] * (xx.pow(3) - 15 * xx.square() * yy + 15 * xx * yy.square() - yy.pow(3)),
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
