from math import ceil

import torch

from ....splats.camera import PinholeCamera
from ....splats.splats import GaussianSplat
from ..reference import emit_intersections, project_and_count, sort_intersections

IMAGE_SIZE = (32, 32)
TILE_SIZE = 8
NEAR = 0.1
FAR = 100.0
CONFIDENCE = 0.95
OPACITY_THRESHOLD = 0.999

PCT_PER_EDGE_CASE = 0.05


def make_scene(num_points: int = 1_000, sh_degree: int = 1) -> tuple[GaussianSplat, PinholeCamera]:
    """Create a seeded scene with coverage of rasterization edge cases."""

    M = int(num_points * PCT_PER_EDGE_CASE)  # Num points for each edge case
    if M <= 0:
        raise ValueError(
            f"num_points must be at least {ceil(1 / PCT_PER_EDGE_CASE)}. Got {num_points}"
        )

    with torch.random.fork_rng():
        torch.manual_seed(42)
        splats = GaussianSplat(num_points, sh_degree=sh_degree)
        with torch.no_grad():
            splats.mean[:, :2].uniform_(-2.5, 2.5)
            splats.mean[:, 2].uniform_(1.0, 8.0)

            splats.mean[:M, 2].uniform_(0.03, NEAR)  # Near-plane clipping
            splats.mean[M : 2 * M, 2].uniform_(FAR, FAR + 20.0)  # Far-plane clipping
            splats.mean[2 * M : 3 * M, :2].uniform_(8.0, 20.0)  # Off-screen clipping

            splats.scale.uniform_(-3.0, -1.0)
            splats.scale[3 * M : 4 * M].uniform_(-0.5, 0.25)  # Large Gaussians

            splats.opacity.uniform_(-3.0, 3.0)
            splats.color.normal_(0.0, 0.5)

    camera = PinholeCamera(
        image_size=IMAGE_SIZE,
        focal_length=(24.0, 24.0),
        principal_point=(IMAGE_SIZE[0] / 2, IMAGE_SIZE[1] / 2),
    )
    return splats, camera


def kernel_arguments(
    splats: GaussianSplat, camera: PinholeCamera, rendering_mode: str = "gaussian"
) -> tuple:
    fx, fy = camera.focal_length
    cx, cy = camera.principal_point
    width, height = camera.image_size
    return (
        splats.mean.detach(),
        splats.rotation.detach(),
        splats.scale.detach(),
        splats.opacity.detach(),
        splats.color.detach(),
        camera.rotation_matrix,
        camera.translation,
        fx,
        fy,
        cx,
        cy,
        width,
        height,
        TILE_SIZE,
        NEAR,
        FAR,
        rendering_mode,
        CONFIDENCE,
    )


def make_projected_case() -> tuple:
    splats, camera = make_scene(100)
    projected = project_and_count(*kernel_arguments(splats, camera))
    mean, inverse_covariance, color, opacity, bounds, counts = projected
    offsets = counts.cumsum(0) - counts
    keys, gaussian_ids = emit_intersections(
        bounds, mean[:, 2], counts, offsets, IMAGE_SIZE[0] // TILE_SIZE
    )
    gaussian_ids, ranges = sort_intersections(
        keys,
        gaussian_ids,
        IMAGE_SIZE[0] // TILE_SIZE * (IMAGE_SIZE[1] // TILE_SIZE),
    )
    return mean, inverse_covariance, color, opacity, gaussian_ids, ranges
