import math

import torch

SH_C0 = 0.28209479177387814


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
    directions: torch.Tensor,  # N x 3, unit vectors
    num_coefficients: int,
) -> torch.Tensor:  # N x (degree + 1)^2
    """Normalized real SH, ordered by band l and order m = -l, ..., l."""
    degree = math.isqrt(num_coefficients) - 1
    if (degree + 1) ** 2 != num_coefficients or not 0 <= degree <= 12:
        raise ValueError(f"Unsupported spherical-harmonic coefficient count: {num_coefficients}")

    x, y, z = directions.unbind(dim=-1)
    bands = [torch.full_like(z[:, None], SH_C0)]
    previous = bands[0]  # N x (2l - 1)
    older = torch.zeros_like(previous)
    real, imaginary = previous[:, 0], torch.zeros_like(z)

    # Recur over bands, evaluating all directions and orders in parallel.
    for band in range(1, degree + 1):
        order = torch.arange(1 - band, band, device=z.device, dtype=z.dtype)
        scale = torch.sqrt((4 * band**2 - 1) / (band**2 - order.square()))
        correction = torch.sqrt(((band - 1) ** 2 - order.square()) / (4 * (band - 1) ** 2 - 1))
        interior = scale * (z[:, None] * previous - correction * older)

        # The two new edge orders come from multiplying by x + iy.
        # Band 1 introduces the sqrt(2) normalization for nonzero orders.
        edge_scale = -math.sqrt(3 if band == 1 else (2 * band + 1) / (2 * band))
        real, imaginary = (
            edge_scale * (x * real - y * imaginary),
            edge_scale * (x * imaginary + y * real),
        )
        current = torch.cat((imaginary[:, None], interior, real[:, None]), dim=-1)
        bands.append(current)
        older = torch.nn.functional.pad(previous, (1, 1))
        previous = current

    return torch.cat(bands, dim=-1)


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
