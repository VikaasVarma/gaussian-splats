import torch

from ..interface import rasterize
from ..torch import rasterize as torch_rasterize
from .common import (
    CONFIDENCE,
    FAR,
    NEAR,
    OPACITY_THRESHOLD,
    TILE_SIZE,
    make_scene,
)


def test_matches_torch() -> None:
    splats, camera = make_scene()
    expected = torch_rasterize(
        splats, camera, TILE_SIZE, OPACITY_THRESHOLD, NEAR, FAR, confidence=CONFIDENCE
    )
    actual = rasterize(
        splats, camera, TILE_SIZE, OPACITY_THRESHOLD, NEAR, FAR, confidence=CONFIDENCE
    )
    assert actual.device == splats.mean.device
    assert actual.dtype == splats.mean.dtype
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-4)


def test_backward_matches_torch_reference() -> None:
    triton_splats, triton_camera = make_scene(20)
    torch_splats, torch_camera = make_scene(20)

    triton_params = (
        triton_splats.mean,
        triton_splats.rotation,
        triton_splats.scale,
        triton_splats.opacity,
        triton_splats.color,
    )
    torch_params = (
        torch_splats.mean,
        torch_splats.rotation,
        torch_splats.scale,
        torch_splats.opacity,
        torch_splats.color,
    )

    for p in (*triton_params, *torch_params):
        p.requires_grad_(True)

    triton_out = rasterize(triton_splats, triton_camera).square().mean()
    triton_out.backward()
    triton_grads = [p.grad.detach().clone() for p in triton_params]

    torch_out = torch_rasterize(torch_splats, torch_camera).square().mean()
    torch_out.backward()
    torch_grads = [p.grad.detach().clone() for p in torch_params]

    for triton_grad, torch_grad in zip(triton_grads, torch_grads, strict=True):
        torch.testing.assert_close(triton_grad, torch_grad, rtol=2e-3, atol=2e-4)
