from __future__ import annotations

import torch
import torch.nn.functional as F

from gaussian_splats.splats.camera import PinholeCamera

from ..scene import Scene


def light_rays(
    scene: Scene, origins: torch.Tensor
) -> tuple[
    torch.Tensor,  # N x L x 3
    torch.Tensor,  # L x 3
    torch.Tensor,  # N x L
]:
    N, L = len(origins), len(scene.lights.positions)
    positions, directions, types = (
        scene.lights.positions,
        scene.lights.directions,
        scene.lights.types.bool(),
    )

    point_offsets = positions.view(1, L, 3) - origins.view(N, 1, 3)
    point_distances = torch.linalg.vector_norm(point_offsets, dim=-1)
    point_directions = F.normalize(point_offsets, dim=-1)
    directions = torch.where(types.view(1, L, 1), point_directions, directions.view(1, L, 3))
    colors = scene.lights.colors * scene.lights.energies.view(L, 1)
    distances = torch.where(types.view(1, L), point_distances, torch.inf)

    return directions, colors, distances


def generate_pinhole_rays(
    camera: PinholeCamera, width: int, height: int, device: torch.device
) -> tuple[
    torch.Tensor,  # N x 3 (origin)
    torch.Tensor,  # N x 3 (direction)
]:
    dtype = camera.translation.dtype
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + 0.5,
        torch.arange(width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    fx, fy = camera.focal_length
    camera_directions = torch.stack(
        (
            (x - width / 2) / fx,
            (height / 2 - y) / fy,
            -torch.ones_like(x),
        ),
        dim=-1,
    )
    camera_directions = F.normalize(camera_directions, dim=-1)
    rotation = camera.rotation_matrix.to(device)
    origins = camera.position.to(device).expand(width * height, 3)
    directions = camera_directions.reshape(-1, 3) @ rotation
    return origins, directions


@torch.no_grad()
def intersection_from_point(
    scene: Scene,
    origin: torch.Tensor,  # N x 3
    directions: torch.Tensor,  # N x 3
) -> tuple[
    torch.Tensor,  # N (hit)
    torch.Tensor,  # N (distance)
    torch.Tensor,  # N (triangle index)
    torch.Tensor,  # N x 2 (barycentric coordinates)
]:
    N = len(directions)
    triangles = scene.vertices  # T x 3 x 3
    v0 = triangles[:, 0]  # T x 3
    e0 = triangles[:, 1] - v0  # T x 3
    e1 = triangles[:, 2] - v0  # T x 3

    if len(triangles) == 0 or len(directions) == 0:
        misses = directions.new_full((N,), float("inf"))
        return (
            torch.zeros(N, dtype=torch.bool, device=directions.device),
            misses,
            torch.full_like(misses, -1, dtype=torch.long),
            directions.new_zeros(N, 2),
        )

    p = torch.linalg.cross(directions[:, None], e1[None])  # N x T x 3
    determinant = torch.einsum("ti,nti->nt", e0, p)  # N x T
    inverse = determinant.reciprocal()  # N x T

    tvec = origin[:, None] - v0[None]  # N x T x 3
    u = torch.einsum("nti,nti->nt", tvec, p) * inverse  # N x T
    qvec = torch.linalg.cross(tvec, e0[None])  # N x T x 3
    v = torch.einsum("ni,nti->nt", directions, qvec) * inverse  # N x T
    t = torch.einsum("ti,nti->nt", e1, qvec) * inverse  # N x T

    valid = (determinant.abs() > 1e-8) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
    distance, triangle = torch.where(valid, t, float("inf")).min(dim=1)
    hit = distance.isfinite()

    index = triangle.view(N, 1)
    determinant = determinant.gather(1, index).view(N)
    u = u.gather(1, index).view(N)
    v = v.gather(1, index).view(N)
    uv = torch.stack((u, v), dim=-1)
    uv = torch.where(hit[:, None], uv, 0)

    return hit, distance, torch.where(hit, triangle, -1), uv


@torch.no_grad()
def occluded(
    scene: Scene,
    origins: torch.Tensor,  # N x 3 or N x L x 3
    directions: torch.Tensor,  # N x L x 3
    distances: torch.Tensor,  # N x L
    epsilon: float = 1e-5,
) -> torch.Tensor:
    N, L = directions.shape[:2]
    R = N * L
    triangles = scene.vertices
    v0 = triangles[:, 0]
    e0 = triangles[:, 1] - v0
    e2 = triangles[:, 2] - v0

    if origins.ndim == 2:
        origins = origins.view(N, 1, 3).expand(N, L, 3).contiguous()

    origins = origins.view(R, 3)
    directions = F.normalize(directions, dim=-1).view(R, 3)
    distances = distances.view(R, 1)

    p = torch.linalg.cross(directions[:, None], e2[None])  # R x T x 3
    determinant = torch.einsum("ti,rti->rt", e0, p)  # R x T
    inverse = determinant.where(determinant.abs() > 1e-6, determinant.new_ones(())).reciprocal()
    t = origins[:, None] - v0[None]  # R x T x 3
    u = torch.einsum("rti,rti->rt", t, p) * inverse  # R x T
    q = torch.linalg.cross(t, e0[None])  # R x T x 3
    v = torch.einsum("ri,rti->rt", directions, q) * inverse  # R x T
    t = torch.einsum("ti,rti->rt", e2, q) * inverse  # R x T

    valid = (
        (determinant.abs() > epsilon)
        & (u > epsilon)
        & (v > epsilon)
        & (u + v < 1 - epsilon)
        & (t > 0)
        & (t < distances)
    )
    return valid.any(-1).view(N, L)
