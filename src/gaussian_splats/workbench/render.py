from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.func import vmap

from gaussian_splats.blender import Scene, linear_to_srgb
from gaussian_splats.splats.camera import PinholeCamera

from .shading import SHADOW_DIRECTION, shade


@torch.no_grad()
def intersect_from_point(
    scene: Scene,
    origin: torch.Tensor,
    directions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    triangles = scene.vertices
    v0 = triangles[:, 0]
    e1 = triangles[:, 1] - v0
    e2 = triangles[:, 2] - v0

    tvec = origin - v0
    qvec = torch.linalg.cross(tvec, e1)
    coefficients = torch.stack(  # T x 3 x 3
        (
            torch.linalg.cross(e2, e1),
            torch.linalg.cross(e2, tvec),
            qvec,
        ),
        dim=1,
    )
    distances = (e2 * qvec).sum(-1)  # T

    if len(triangles) == 0 or len(directions) == 0:
        misses = directions.new_full((len(directions),), float("inf"))
        return (
            torch.zeros(len(directions), dtype=torch.bool, device=directions.device),
            misses,
            torch.full_like(misses, -1, dtype=torch.long),
            directions.new_zeros(len(directions), 2),
        )

    @torch.compile(dynamic=True)
    def closest_hit(directions, coefficients, distances):
        def solve(d, M):
            return M[:, 0] * d[0] + M[:, 1] * d[1] + M[:, 2] * d[2]

        def hit_distance(d, M, s):
            det, u, v = solve(d, M).unbind()
            inverse = det.reciprocal()
            u, v, t = u * inverse, v * inverse, s * inverse
            valid = (det.abs() > 1e-8) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
            return torch.where(valid, t, float("inf"))

        def first_hit(d, M, s):
            t, triangle = vmap(hit_distance, in_dims=(None, 0, 0))(d, M, s).min(0)
            return t, triangle

        t, triangle = vmap(first_hit, in_dims=(0, None, None))(directions, coefficients, distances)

        det, u, v = vmap(solve)(directions, coefficients[triangle]).unbind(-1)
        hit = t.isfinite()
        uv = torch.where(hit[:, None], torch.stack((u, v), dim=-1) / det[:, None], 0)
        return hit, t, torch.where(hit, triangle, -1), uv

    return closest_hit(directions, coefficients, distances)


@torch.no_grad()
def occluded(
    scene: Scene,
    origins: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    center = scene.scene_bounds.mean(0)
    triangles = scene.vertices - center
    v0 = triangles[:, 0]
    e1 = triangles[:, 1] - v0
    e2 = triangles[:, 2] - v0

    direction = F.normalize(direction, dim=-1).expand_as(e1)
    pvec = torch.linalg.cross(direction, e2)
    determinant = (e1 * pvec).sum(-1)
    inverse = determinant.reciprocal()[:, None]
    coefficients = torch.stack(  # T x 3 x 3
        (
            pvec * inverse,
            torch.linalg.cross(e1, direction) * inverse,
            torch.linalg.cross(e1, e2) * inverse,
        ),
        dim=1,
    )
    planes = torch.cat(  # T x 3 x 4
        (coefficients, -(coefficients * v0[:, None]).sum(-1, keepdim=True)), dim=-1
    )

    # Give degenerate triangles t = -1 (guaranteed miss)
    never = planes.new_tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -1]])
    planes = torch.where(determinant.abs().gt(1e-8)[:, None, None], planes, never)

    if len(triangles) == 0 or len(origins) == 0:
        return origins.new_zeros(len(origins), dtype=torch.bool)

    @torch.compile(dynamic=True)
    def any_hit(origins, planes):
        def hit(o, P):
            u, v, t = (P[:, 0] * o[0] + P[:, 1] * o[1] + P[:, 2] * o[2] + P[:, 3]).unbind()
            return (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)

        def blocked(o, P):
            return vmap(hit, in_dims=(None, 0))(o, P).any()

        return vmap(blocked, in_dims=(0, None))(origins, planes)

    return any_hit(origins - center, planes)


def _camera_rays(
    camera: PinholeCamera, width: int, height: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = camera.translation.dtype
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + 0.5,
        torch.arange(width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    fx, fy = camera.focal_length
    camera_directions = torch.stack(
        ((x - width / 2) / fx, (height / 2 - y) / fy, torch.ones_like(x)), dim=-1
    )
    camera_directions = F.normalize(camera_directions, dim=-1)
    rotation = camera.rotation_matrix.to(device)
    position = camera.position.to(device)
    directions = camera_directions.reshape(-1, 3) @ rotation
    return position, directions


@torch.no_grad()
def render(
    scene: Scene,
    camera: PinholeCamera,
    width: int | None = None,
    height: int | None = None,
    *,
    color_mode: str = "material",
    lighting: str = "studio",
    background: tuple[float, float, float, float] = (0, 0, 0, 0),
    srgb: bool = True,
) -> torch.Tensor:
    width, height = width or camera.image_size[0], height or camera.image_size[1]
    origin, directions = _camera_rays(camera, width, height, scene.vertices.device)

    hit, _, triangle_ids, uv = intersect_from_point(scene, origin, directions)

    triangle_ids, uv = triangle_ids[hit], uv[hit]
    triangles = scene.vertices[triangle_ids]
    indices = hit.nonzero().squeeze(-1)

    barycentric = torch.cat((1 - uv.sum(-1, keepdim=True), uv), dim=-1)
    positions = (triangles * barycentric[..., None]).sum(1)
    normals = F.normalize((scene.normals[triangle_ids] * barycentric[..., None]).sum(1), dim=-1)

    color, alpha, _ = query_radiance(
        scene,
        positions,
        normals,
        triangle_ids,
        barycentric,
        eps=1e-4,
        color_mode=color_mode,
        lighting=lighting,
    )

    result = directions.new_tensor(background).expand(len(directions), 4).contiguous()
    result[indices, :3] = (linear_to_srgb(color[:, :3]) if srgb else color[:, :3]).clamp(0, 1)
    result[indices, 3] = alpha
    return result.reshape(height, width, 4)


@torch.no_grad()
def query_radiance(
    scene: Scene,
    points: torch.Tensor,
    normals: torch.Tensor,
    triangle_id: torch.Tensor,
    barycentric: torch.Tensor,
    *,
    eps: float = 1e-4,
    color_mode: str = "material",
    lighting: str = "studio",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    uv = (scene.uvs[triangle_id] * barycentric[..., None]).sum(1)
    material_id = scene.material_id[triangle_id]

    light = SHADOW_DIRECTION.to(device=points.device, dtype=points.dtype)

    shadowed = occluded(scene, points + normals * eps, light)
    color = shade(scene, points, normals, uv, material_id, color_mode, lighting, shadowed)
    alpha = torch.ones(len(points), device=points.device, dtype=points.dtype)
    return color, alpha, points
