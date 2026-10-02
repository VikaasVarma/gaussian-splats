from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Materials:
    colors: torch.Tensor  # M x 4
    speculars: torch.Tensor  # M x 3
    shininesses: torch.Tensor  # M

    texture_rects: torch.Tensor  # M x 4: x, y, width, height
    texture_atlas: torch.Tensor  # H x W x 4

    def to(self, device: torch.device) -> "Materials":
        return Materials(
            colors=self.colors.to(device),
            speculars=self.speculars.to(device),
            shininesses=self.shininesses.to(device),
            texture_rects=self.texture_rects.to(device),
            texture_atlas=self.texture_atlas.to(device),
        )

    def sample(
        self,
        uvs: torch.Tensor,  # N x 2
        material_ids: torch.Tensor,  # T
    ) -> torch.Tensor:  # T x 3
        rects = self.texture_rects[material_ids]
        has_texture = rects[:, 0] >= 0

        x, y, width, height = rects.clamp_min(0).float().unbind(-1)
        px = x + uvs[:, 0] * (width - 1)
        py = y + (1 - uvs[:, 1]) * (height - 1)

        Ha, Wa, _ = self.texture_atlas.shape
        atlas = self.texture_atlas.permute(2, 0, 1).view(1, 4, Ha, Wa)
        grid = torch.stack((2 * px / (Wa - 1) - 1, 2 * py / (Ha - 1) - 1), dim=-1).view(1, -1, 1, 2)

        texture = F.grid_sample(atlas, grid, mode="bilinear", align_corners=True)
        texture = texture[0, :, :, 0].T[:, :3]
        texture = torch.where(has_texture[:, None], texture, torch.ones_like(texture))

        return texture


@dataclass(frozen=True)
class Lights:
    positions: torch.Tensor  # L x 3
    directions: torch.Tensor  # L x 3
    colors: torch.Tensor  # L x 3
    energies: torch.Tensor  # L
    types: torch.Tensor  # L: 0 = directional, 1 = positional

    def to(self, device: torch.device) -> "Lights":
        return Lights(
            positions=self.positions.to(device),
            directions=self.directions.to(device),
            colors=self.colors.to(device),
            energies=self.energies.to(device),
            types=self.types.to(device),
        )


@dataclass(frozen=True)
class Scene:
    path: Path
    vertices: torch.Tensor  # T x 3 x 3
    normals: torch.Tensor  # T x 3 x 3
    #
    mesh_ids: torch.Tensor  # T
    #
    uvs: torch.Tensor  # T x 3 x 2
    material_ids: torch.Tensor  # T
    materials: Materials
    #
    background_color: torch.Tensor  # 3
    #
    lights: Lights

    def to(self, device: torch.device) -> "Scene":
        return Scene(
            path=self.path,
            vertices=self.vertices.to(device),
            normals=self.normals.to(device),
            mesh_ids=self.mesh_ids.to(device),
            uvs=self.uvs.to(device),
            material_ids=self.material_ids.to(device),
            materials=self.materials.to(device),
            background_color=self.background_color.to(device),
            lights=self.lights.to(device),
        )

    @cached_property
    def mesh_bounds(self) -> torch.Tensor:
        n_meshes = int(self.mesh_ids.max()) + 1
        points = self.vertices.view(-1, 3)
        index = self.mesh_ids.view(-1, 1, 1).expand_as(self.vertices).view(-1, 3)
        minimum = points.new_full((n_meshes, 3), float("inf"))
        maximum = points.new_full((n_meshes, 3), -float("inf"))
        minimum.scatter_reduce_(0, index, points, reduce="amin")
        maximum.scatter_reduce_(0, index, points, reduce="amax")
        return torch.stack((minimum, maximum), dim=1)

    @cached_property
    def scene_bounds(self) -> torch.Tensor:
        points = self.vertices.view(-1, 3)
        return torch.stack((points.amin(0), points.amax(0)))

    @cached_property
    def surface_areas(self) -> torch.Tensor:
        return torch.linalg.vector_norm(
            torch.linalg.cross(
                self.vertices[:, 1] - self.vertices[:, 0],
                self.vertices[:, 2] - self.vertices[:, 0],
            ),
            dim=-1,
        ).mul(0.5)
