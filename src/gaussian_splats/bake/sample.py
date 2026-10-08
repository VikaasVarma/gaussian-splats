import math

import torch
import torch.nn.functional as F

from gaussian_splats.splats.splats import GaussianSplat

from .scene import Scene

PHI = (1 + 5**0.5) / 2
PI = math.pi


def count_samples_per_triangle(scene: Scene, n_splats: int) -> torch.Tensor:
    surface_area = scene.surface_areas.sum()
    raw_counts = scene.surface_areas / surface_area * n_splats
    counts = raw_counts.floor().int()

    remainder = n_splats - int(counts.sum())
    if remainder > 0:  # Allocate remaining splats to largest triangles
        counts[torch.topk(raw_counts - counts, remainder, sorted=False).indices] += 1

    return counts


def sample_splats(
    scene: Scene,
    resolution: int | None = None,
    n_splats: int | None = None,
    sigma: float = 0.65,
) -> tuple[GaussianSplat, torch.Tensor, torch.Tensor]:
    assert not (resolution is None == n_splats is None), (
        "Please provide exactly one of resolution or n_splats"
    )

    surface_area = float(scene.surface_areas.sum().item())
    extent = float((scene.scene_bounds[1] - scene.scene_bounds[0]).max().item())

    if n_splats is None:
        n_splats = max(1, int(round(surface_area * resolution**2 / extent**2)))
        print(f"For resolution {resolution}, sampling {n_splats:,} splats")

    if resolution is None:
        resolution = extent * (n_splats / surface_area) ** 0.5
        print(f"Sampling {n_splats:,} splats -> Equivalent Resolution: {resolution:.3f}")

    # Generate number of samples per triangle
    triangles = scene.vertices
    device = triangles.device
    counts = count_samples_per_triangle(scene, n_splats)

    counts, triangle_id, offsets = torch.repeat_interleave(  # Vectorize all interleavings
        torch.stack(
            (
                counts,
                torch.arange(len(counts), device=device),
                counts.cumsum(0) - counts,
            )
        ),
        counts,
        dim=-1,
        output_size=n_splats,
    )
    local_id = torch.arange(n_splats, device=device) - offsets

    # Fill each triangle with a deterministic 2D grid in barycentric UV space.
    rows = counts.float().sqrt().ceil().long()
    columns = (counts + rows - 1) // rows
    row = local_id // columns
    column = local_id % columns
    u = (row + 0.5) / rows
    v = (column + 0.5) / columns * (1.0 - u)
    barycentric = torch.stack((1.0 - u - v, u, v), dim=-1)

    # Compute GS Parameters
    positions = (barycentric[..., None] * triangles[triangle_id]).sum(dim=1)

    # TODO: Handle missing normals
    normals = F.normalize(scene.normals, dim=-1)
    normals = F.normalize((barycentric[..., None] * normals[triangle_id]).sum(dim=1), dim=-1)

    x, y, z = normals.unbind(dim=-1)
    rotation = torch.stack((1.0 + z, -y, x, torch.zeros_like(z)), dim=-1)
    near_neg_z = rotation.norm(dim=-1) < 1e-6  # Fallback case
    rotation[near_neg_z] = rotation.new_tensor((0.0, 1.0, 0.0, 0.0))
    rotation = F.normalize(rotation, dim=-1)

    spacing = extent / resolution if resolution is not None else (surface_area / n_splats) ** 0.5
    scale = positions.new_tensor((spacing * sigma, spacing * sigma, 1e-7)).log()
    opacity = positions.new_full((n_splats, 1), math.log(1e6))
    color = positions.new_zeros((n_splats, 1, 3))

    return (
        GaussianSplat.from_tensors(
            mean=positions,
            rotation=rotation,
            scale=scale.expand_as(positions),
            opacity=opacity,
            color=color,
            normals=normals,
        ),
        triangle_id,
        barycentric,
    )
