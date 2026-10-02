from __future__ import annotations

import torch
import torch.nn.functional as F

from gaussian_splats.splats.camera import PinholeCamera

from ..blender import linear_to_srgb
from ..scene import Scene
from .types import RayBackend, RenderBackend
from .utils import generate_pinhole_rays, intersection_from_point, occluded

STUDIO_DIRECTIONS = torch.tensor(
    (
        (-0.854701, 0.111111, 0.507091),
        (0.058607, -0.987943, -0.143295),
        (0.972202, 0.075846, -0.221518),
    )
)
STUDIO_COLORS = torch.tensor(
    (
        (0.723042, 0.723042, 0.723042),
        (0.063100, 0.069978, 0.067951),
        (0.157432, 0.163405, 0.214035),
    )
)


class WorkbenchBackend(RayBackend, RenderBackend):
    """
    A torch approximation of the blender workbench renderer.
    """

    def __init__(
        self,
        ambient_strength: float = 0.05,
        diffuse_strength: float = 1.0,
    ) -> None:
        self.ambient_strength = ambient_strength
        self.diffuse_strength = diffuse_strength

    @torch.no_grad()
    def query(
        self,
        scene: Scene,
        origins: torch.Tensor,  # N x 3
        directions: torch.Tensor,  # N x 3
        triangle_ids: torch.Tensor,  # N
        barycentric: torch.Tensor,  # N x 3
        studio_directions: torch.Tensor | None = None,  # N x 3
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        material_ids = scene.material_ids[triangle_ids]
        material_color = scene.materials.colors[material_ids]
        uvs = torch.einsum("ni,nij->nj", barycentric, scene.uvs[triangle_ids])
        texture_color = scene.materials.sample(uvs, material_ids)

        base_color = material_color.clone()
        base_color[:, :3] *= texture_color

        normals = F.normalize(
            torch.einsum("ni,nij->nj", barycentric, scene.normals[triangle_ids]), dim=-1
        )
        studio_directions = (
            studio_directions.to(origins)
            if studio_directions is not None
            else STUDIO_DIRECTIONS.to(origins)
        )
        studio_colors = STUDIO_COLORS.to(origins)

        shaded = occluded(
            scene,
            origins + normals * 1e-4,
            studio_directions.expand(len(origins), -1, -1).contiguous(),
            origins.new_full((len(origins), len(studio_directions)), float("inf")),
        )

        # Ambient
        ambient = base_color[:, :3]

        # Diffuse
        strength = (normals @ studio_directions.mT).clamp_min(0)
        strength = strength * ~shaded
        diffuse = base_color[:, :3] * (strength @ studio_colors)

        # Combine
        color = (self.ambient_strength * ambient + self.diffuse_strength * diffuse).clamp(0, 1)

        rgba = torch.cat((color, base_color[:, 3:4]), dim=-1)
        valid = torch.ones_like(triangle_ids, dtype=torch.bool)
        return rgba, valid

    @torch.no_grad()
    def render(
        self,
        scene: Scene,
        camera: PinholeCamera,
        width: int,
        height: int,
        **kwargs: object,
    ) -> torch.Tensor:
        origins, directions = generate_pinhole_rays(camera, width, height, scene.vertices.device)
        hit, _, triangle_ids, uv = intersection_from_point(scene, origins, directions)

        image = scene.background_color.view(1, 3).expand(width * height, 3).clone()
        barycentric = torch.cat((1 - uv.sum(1, keepdim=True), uv), dim=1)
        positions = torch.einsum("ni,nij->nj", barycentric[hit], scene.vertices[triangle_ids[hit]])
        color, _ = self.query(
            scene,
            positions,
            directions[hit],
            triangle_ids[hit],
            barycentric[hit],
        )
        image[hit] = color[:, :3]
        return linear_to_srgb(image.view(height, width, 3)).clamp(0, 1)


__all__ = ["WorkbenchBackend"]
