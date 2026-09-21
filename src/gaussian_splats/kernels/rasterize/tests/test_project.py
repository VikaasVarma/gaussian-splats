import pytest
import torch

from ..project import project_and_count as triton_project_and_count
from ..reference import project_and_count as torch_project_and_count
from .common import kernel_arguments, make_scene


def _assert_matches_torch(arguments: tuple) -> None:
    expected = torch_project_and_count(*arguments)
    actual = triton_project_and_count(*arguments)

    for result, reference in zip(actual[:4], expected[:4], strict=True):
        torch.testing.assert_close(result, reference, rtol=2e-4, atol=2e-4)

    torch.testing.assert_close(actual[4], expected[4], rtol=0, atol=0)
    torch.testing.assert_close(actual[5], expected[5], rtol=0, atol=0)


@pytest.mark.parametrize("sh_degree", (0, 4), ids=("sh0", "sh4"))
def test_matches_torch_reference(sh_degree: int) -> None:
    splats, camera = make_scene(sh_degree=sh_degree)
    arguments = kernel_arguments(splats, camera)
    _assert_matches_torch(arguments)
