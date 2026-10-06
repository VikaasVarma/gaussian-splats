from __future__ import annotations

import torch
import torch.nn.functional as F

from gaussian_splats.splats.camera import PinholeCamera

from ..blender import linear_to_srgb
from ..scene import Scene
from .types import RayBackend, RenderBackend
from .utils import generate_pinhole_rays, intersection_from_point, light_rays, occluded

AMBIENT_STRENGTH = 0.25
DIFFUSE_STRENGTH = 1.0
SPECULAR_STRENGTH = 1.0


class PhongBackend(RayBackend, RenderBackend):
    def __init__(
        self,
        ambient_strength: float = AMBIENT_STRENGTH,
        diffuse_strength: float = DIFFUSE_STRENGTH,
        specular_strength: float = SPECULAR_STRENGTH,
    ) -> None:
        self.ambient_strength = ambient_strength
        self.diffuse_strength = diffuse_strength
        self.specular_strength = specular_strength

    def query(
        self,
        scene: Scene,
        origins: torch.Tensor,  # N x 3
        directions: torch.Tensor,  # N x 3
        #
        triangle_ids: torch.Tensor,  # N
        barycentric: torch.Tensor,  # N x 3
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N = len(origins)

        uvs = torch.einsum("ni,nij->nj", barycentric, scene.uvs[triangle_ids])
        material_ids = scene.material_ids[triangle_ids]
        texture_color = scene.materials.sample(uvs, material_ids)
        material_color, specular, shininess = (
            scene.materials.colors[material_ids],
            scene.materials.speculars[material_ids],
            scene.materials.shininesses[material_ids],
        )

        base_color = material_color.clone()
        base_color[:, :3] *= texture_color

        normals = F.normalize(
            torch.einsum("ni,nij->nj", barycentric, scene.normals[triangle_ids]), dim=-1
        )

        scene_scale = torch.norm(scene.scene_bounds[1] - scene.scene_bounds[0])
        epsilon = max(float(scene_scale) * 1e-4, 1e-5)

        light_directions, light_colors, light_distances = light_rays(scene, origins)
        L = light_directions.shape[1]
        shaded = occluded(
            scene,
            origins + normals * epsilon,
            light_directions,
            light_distances,
        )

        # Ambient
        ambient = base_color[:, :3]

        # Diffuse
        strength = torch.einsum("ni,nli->nl", normals, light_directions).clamp_min(0)
        strength = strength * ~shaded
        diffuse = base_color[:, :3] * (strength @ light_colors)

        # Specular
        reflection = 2 * strength.view(N, L, 1) * normals.view(N, 1, 3) - light_directions
        highlight = (
            torch.einsum("nli,ni->nl", reflection, directions)
            .clamp_min(0)
            .pow(shininess.view(N, 1))
        )
        highlight = torch.where(strength > 0, highlight, 0)
        specular = specular * (highlight @ light_colors)

        # Combine
        color = (
            self.ambient_strength * ambient
            + self.diffuse_strength * diffuse
            + self.specular_strength * specular
        ).clamp(0, 1)
        rgba = torch.cat((color, base_color[:, 3:4]), dim=-1)
        valid = torch.ones_like(triangle_ids, dtype=torch.bool)
        return rgba, valid

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

        image = scene.background_color.view(1, 3).expand(width * height, 3).contiguous()
        barycentric = torch.cat((1 - uv.sum(1, keepdim=True), uv), dim=1)
        positions = torch.einsum("ni,nij->nj", barycentric[hit], scene.vertices[triangle_ids[hit]])
        color, _ = self.query(
            scene,
            positions,
            -directions[hit],
            triangle_ids[hit],
            barycentric[hit],
        )
        image[hit] = color[:, :3]
        return linear_to_srgb(image.view(height, width, 3)).clamp(0, 1)
