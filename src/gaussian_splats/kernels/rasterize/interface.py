"""Public interface for the Triton rasterizer."""

import os
from typing import Literal

import torch

from ...splats.camera import Camera
from ...splats.splats import GaussianSplat
from .composite import composite
from .composite_bwd import composite_bwd
from .intersections import emit_intersections, sort_intersections
from .project import project_and_count
from .project_bwd import project_bwd


@torch.library.custom_op("gaussian_splats::rasterize", mutates_args=())
def _rasterize(
    mean: torch.Tensor,
    rotation: torch.Tensor,
    scale: torch.Tensor,
    opacity: torch.Tensor,
    color: torch.Tensor,
    camera_rotation: torch.Tensor,
    camera_translation: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_width: int,
    image_height: int,
    tile_size: int,
    opacity_threshold: float,
    near: float,
    far: float,
    rendering_mode: str,
    confidence: float,
) -> tuple[
    torch.Tensor,  # image
    torch.Tensor,  # final_transmittance
    torch.Tensor,  # n_contributors
    torch.Tensor,  # projected_mean
    torch.Tensor,  # inverse_covariance
    torch.Tensor,  # projected_color
    torch.Tensor,  # projected_opacity
    torch.Tensor,  # gaussian_ids
    torch.Tensor,  # ranges
    torch.Tensor,  # counts
]:
    (projected_mean, inverse_covariance, projected_color, projected_opacity, bounds, counts) = (
        project_and_count(
            mean=mean,
            rotation=rotation,
            scale=scale,
            opacity=opacity,
            color=color,
            camera_rotation=camera_rotation,
            camera_translation=camera_translation,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            image_width=image_width,
            image_height=image_height,
            tile_size=tile_size,
            near=near,
            far=far,
            rendering_mode=rendering_mode,
            confidence=confidence,
        )
    )

    keys, gaussian_ids = emit_intersections(
        bounds=bounds,
        depths=projected_mean[:, 2],
        counts=counts,
        tiles_w=image_width // tile_size,
    )
    gaussian_ids, ranges = sort_intersections(
        keys=keys,
        gaussian_ids=gaussian_ids,
        num_tiles=(image_width // tile_size) * (image_height // tile_size),
    )

    image, final_transmittance, n_contributors = composite(
        mean=projected_mean,
        inverse_covariance=inverse_covariance,
        color=projected_color,
        opacity=projected_opacity,
        gaussian_ids=gaussian_ids,
        ranges=ranges,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        opacity_threshold=opacity_threshold,
        rendering_mode=rendering_mode,
        confidence=confidence,
    )

    return (
        image,
        final_transmittance,
        n_contributors,
        projected_mean,
        inverse_covariance,
        projected_color,
        projected_opacity,
        gaussian_ids,
        ranges,
        counts,
    )


@_rasterize.register_fake
def _rasterize_fake(
    mean: torch.Tensor,
    rotation: torch.Tensor,
    scale: torch.Tensor,
    opacity: torch.Tensor,
    color: torch.Tensor,
    camera_rotation: torch.Tensor,
    camera_translation: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_width: int,
    image_height: int,
    tile_size: int,
    opacity_threshold: float,
    near: float,
    far: float,
    rendering_mode: str,
    confidence: float,
) -> list[torch.Tensor]:
    dynamic = torch.library.get_ctx().new_dynamic_size()
    return [
        mean.new_empty((image_height, image_width, 3)),
        mean.new_empty((image_height, image_width)),
        torch.empty((image_height, image_width), dtype=torch.int32, device=mean.device),
        mean.new_empty(mean.shape),
        mean.new_empty((mean.shape[0], 2, 2)),
        mean.new_empty((mean.shape[0], 3)),
        mean.new_empty((mean.shape[0], 1)),
        torch.empty((dynamic,), dtype=torch.int32, device=mean.device),
        torch.empty(
            (
                (image_height // tile_size) * (image_width // tile_size),
                2,
            ),
            dtype=torch.int32,
            device=mean.device,
        ),
        torch.empty((mean.shape[0],), dtype=torch.int32, device=mean.device),
    ]


def _rasterize_setup_context(ctx, inputs, output) -> None:
    ctx.save_for_backward(*inputs[:7], *output[1:])
    (
        ctx.fx,
        ctx.fy,
        ctx.cx,
        ctx.cy,
        ctx.image_width,
        ctx.image_height,
        ctx.tile_size,
        ctx.opacity_threshold,
        ctx.near,
        ctx.far,
        ctx.rendering_mode,
        ctx.confidence,
    ) = inputs[7:]


def _rasterize_backward(
    ctx,
    *grad_outputs: torch.Tensor | None,
) -> tuple[torch.Tensor | None, ...]:
    # This shit is AI generated and unverified

    """Run the reverse compositor and finish projection gradients."""
    grad_output = grad_outputs[0]
    assert ctx.rendering_mode == "gaussian", (
        "Backward gradients are only supported for Gaussian rendering."
    )
    inputs = ctx.saved_tensors[:7]
    outputs = ctx.saved_tensors[7:]
    mean, rotation, scale, opacity, color, camera_rotation, camera_translation = inputs

    (
        final_transmittance,
        n_contributors,
        projected_mean,
        inverse_covariance,
        projected_color,
        projected_opacity,
        gaussian_ids,
        ranges,
        counts,
    ) = outputs

    composite_bwd_result = composite_bwd(
        projected_mean,
        inverse_covariance,
        projected_color,
        projected_opacity,
        gaussian_ids,
        ranges,
        grad_output,
        final_transmittance,
        n_contributors,
        ctx.image_width,
        ctx.image_height,
        ctx.tile_size,
        ctx.opacity_threshold,
        ctx.rendering_mode,
        atomic=os.environ.get("GAUSSIAN_SPLATS_ATOMIC_BACKWARD") == "1",
    )
    grad_projected = composite_bwd_result

    gradients = project_bwd(
        mean,
        rotation,
        scale,
        opacity,
        color,
        camera_rotation,
        camera_translation,
        inverse_covariance,
        grad_projected,
        counts,
        ctx.fx,
        ctx.fy,
        ctx.image_width,
        ctx.image_height,
    )

    return (*gradients, *(None,) * (19 - len(gradients)))


torch.library.register_autograd(
    "gaussian_splats::rasterize", _rasterize_backward, setup_context=_rasterize_setup_context
)


def rasterize(
    splats: GaussianSplat,
    camera: Camera,
    tile_size: int = 4,
    opacity_threshold: float = 0.999,
    near: float = 0.2,
    far: float = 100.0,
    rendering_mode: Literal["gaussian", "ellipsoid"] = "gaussian",
    confidence: float = 0.95,
    indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Render splats with the Triton rasterizer."""
    fx, fy = camera.focal_length
    cx, cy = camera.principal_point
    image_width, image_height = camera.image_size

    assert tile_size > 0 and tile_size & (tile_size - 1) == 0
    assert image_width % tile_size == 0 and image_height % tile_size == 0

    mean, rotation, scale, opacity, color = (
        splats.mean,
        splats.rotation,
        splats.scale,
        splats.opacity,
        splats.color,
    )

    if indices is not None:
        mean, rotation, scale, opacity, color = (
            mean[indices],
            rotation[indices],
            scale[indices],
            opacity[indices],
            color[indices],
        )

    return _rasterize(
        mean,
        rotation,
        scale,
        opacity,
        color,
        camera.rotation_matrix,
        camera.translation,
        fx,
        fy,
        cx,
        cy,
        image_width,
        image_height,
        tile_size,
        opacity_threshold,
        near,
        far,
        rendering_mode,
        confidence,
    )[0]
