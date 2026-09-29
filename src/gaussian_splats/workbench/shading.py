from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from gaussian_splats.blender import Scene

STUDIO_DIRECTIONS = torch.tensor(
    (
        (-0.352546, 0.170931, -0.920051),
        (-0.408163, 0.346939, 0.844415),
        (0.521739, 0.826087, 0.212999),
        (0.624519, -0.562067, -0.542269),
    )
)
STUDIO_COLORS = torch.tensor(
    (
        (0.033103, 0.033103, 0.033103),
        (0.521083, 0.538226, 0.538226),
        (0.038403, 0.034357, 0.049529),
        (0.090838, 0.082079, 0.072255),
    )
)
SHADOW_DIRECTION = torch.full((3,), 3**-0.5)


def base_color(
    scene: Scene,
    material_id: torch.Tensor,
    uv: torch.Tensor,
    color_mode: Literal["material", "texture"],
) -> torch.Tensor:
    if color_mode not in {"material", "texture"}:
        raise ValueError(f"Unsupported Workbench color mode: {color_mode}")

    material = scene.material_color[material_id]
    rect = scene.texture_rect[material_id]
    has_texture = rect[:, 0] > 0

    x, y, width, height = rect.clamp_min(0).float().unbind(-1)
    px = x + uv[:, 0] * (width - 1)
    py = y + (1 - uv[:, 1]) * (height - 1)

    atlas = scene.texture_atlas.permute(2, 0, 1)[None]
    atlas_height, atlas_width = scene.texture_atlas.shape[:2]
    grid = torch.stack(
        (
            2 * px / max(atlas_width - 1, 1) - 1,
            2 * py / max(atlas_height - 1, 1) - 1,
        ),
        dim=-1,
    ).view(1, -1, 2, 1)

    texture = F.grid_sample(atlas, grid, mode="bilinear", align_corners=True)
    texture = texture[0, :, :, 0].T
    sampled = texture * material if color_mode == "material" else texture
    return torch.where(has_texture[:, None], sampled, material)


def shade(
    scene: Scene,
    positions: torch.Tensor,
    normals: torch.Tensor,
    uv: torch.Tensor,
    material_id: torch.Tensor,
    color_mode: str = "material",
    lighting: str = "studio",
    shadows: torch.Tensor | None = None,
) -> torch.Tensor:
    color = base_color(scene, material_id, uv, color_mode)

    if lighting == "flat":
        return color
    if lighting != "studio":
        raise ValueError(f"Unsupported Workbench lighting: {lighting}")

    directions = STUDIO_DIRECTIONS.to(device=positions.device, dtype=positions.dtype)
    colors = STUDIO_COLORS.to(device=positions.device, dtype=positions.dtype)
    diffuse = (normals @ directions.T).clamp_min(0)
    result = color[:, :3] * (diffuse @ colors)

    if shadows is not None:
        result = result * torch.where(shadows[:, None], 0.35, 1.0)

    return result.clamp(0, 1)
