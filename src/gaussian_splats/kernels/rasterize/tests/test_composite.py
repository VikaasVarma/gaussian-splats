import torch

from ..composite import composite as triton_composite
from ..reference import composite as torch_composite
from .common import CONFIDENCE, IMAGE_SIZE, OPACITY_THRESHOLD, TILE_SIZE, make_projected_case


def test_matches_torch() -> None:
    inputs = make_projected_case()
    expected = torch_composite(
        *inputs, *IMAGE_SIZE, TILE_SIZE, OPACITY_THRESHOLD, "gaussian", CONFIDENCE
    )
    actual = triton_composite(
        *inputs, *IMAGE_SIZE, TILE_SIZE, OPACITY_THRESHOLD, "gaussian", CONFIDENCE
    )
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_empty_tiles() -> None:
    inputs = list(make_projected_case())
    inputs[-1] = torch.zeros_like(inputs[-1])
    actual = triton_composite(
        *inputs, *IMAGE_SIZE, TILE_SIZE, OPACITY_THRESHOLD, "gaussian", CONFIDENCE
    )
    torch.testing.assert_close(actual, torch.zeros_like(actual))
