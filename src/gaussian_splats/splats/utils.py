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


def evaluate_sh(coefficients: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Evaluate degree 0-3 real spherical harmonics into RGB."""
    x, y, z = directions.unbind(dim=-1)
    xx, yy, zz = x.square(), y.square(), z.square()
    xy, yz, xz = x * y, y * z, x * z
    basis = directions.new_empty((len(directions), coefficients.shape[-1]))
    basis[:, 0] = SH_C0

    if coefficients.shape[-1] > 1:
        basis[:, 1:4] = torch.stack((-SH_C1 * y, SH_C1 * z, -SH_C1 * x), dim=-1)
    if coefficients.shape[-1] > 4:
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
    if coefficients.shape[-1] > 9:
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
    if coefficients.shape[-1] > 16:
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
    if coefficients.shape[-1] > 25:
        raise NotImplementedError(
            f"SH degree {int(coefficients.shape[-1] ** 0.5) - 1} > 4 is not implemented"
        )

    return (torch.einsum("nck,nk->nc", coefficients, basis) + 0.5).clamp(0, 1)


def mahalanobis_distance_squared(
    offset: torch.Tensor,  # M x (tile_size * tile_size) x 2
    inverse_covariance: torch.Tensor,  # M x 2 x 2
) -> torch.Tensor:
    x, y = offset.unbind(dim=-1)
    xx = inverse_covariance[..., None, 0, 0]
    xy = inverse_covariance[..., None, 0, 1]
    yy = inverse_covariance[..., None, 1, 1]
    return xx * x.square() + 2 * xy * x * y + yy * y.square()


def probability(
    distance_squared: torch.Tensor,
    rendering_mode: str = "gaussian",
    confidence: float = 0.95,
) -> torch.Tensor:
    if rendering_mode == "ellipsoid":
        cutoff = -2 * math.log1p(-confidence)
        return (distance_squared <= cutoff).to(distance_squared.dtype)
    return torch.exp(-0.5 * distance_squared)
