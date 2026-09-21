import torch

from ..intersections import emit_intersections, sort_intersections
from ..reference import emit_intersections as torch_emit_intersections
from ..reference import sort_intersections as torch_sort_intersections


def _random_inputs(num_gaussians: int = 256) -> tuple[torch.Tensor, ...]:
    tiles_w = tiles_h = 8
    with torch.random.fork_rng():
        torch.manual_seed(42)
        corners = torch.randint(0, tiles_w + 1, (num_gaussians, 2, 2))
        depths = torch.rand(num_gaussians) * 10.0 + 0.1

    bounds = torch.stack((corners.amin(dim=-1), corners.amax(dim=-1)), dim=-1)
    counts = (bounds[:, :, 1] - bounds[:, :, 0]).prod(dim=-1)
    offsets = counts.cumsum(0) - counts
    return bounds, depths, counts, offsets, tiles_w, tiles_h


def test_emit_matches_torch() -> None:
    bounds, depths, counts, offsets, tiles_w, _ = _random_inputs()
    expected_keys, expected_ids = torch_emit_intersections(bounds, depths, counts, offsets, tiles_w)
    keys, gaussian_ids = emit_intersections(bounds, depths, counts, offsets, tiles_w)
    torch.testing.assert_close(keys, expected_keys, rtol=0, atol=0)
    torch.testing.assert_close(gaussian_ids, expected_ids, rtol=0, atol=0)


def test_sort_and_ranges_match_torch() -> None:
    bounds, depths, counts, offsets, tiles_w, tiles_h = _random_inputs()
    keys, gaussian_ids = emit_intersections(bounds, depths, counts, offsets, tiles_w)
    expected_ids, expected_ranges = torch_sort_intersections(keys, gaussian_ids, tiles_w * tiles_h)
    sorted_ids, ranges = sort_intersections(keys, gaussian_ids, tiles_w * tiles_h)
    torch.testing.assert_close(sorted_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(ranges, expected_ranges, rtol=0, atol=0)
    for start, end in ranges.tolist():
        tile_depths = depths[sorted_ids[start:end]]
        assert bool(torch.all(tile_depths[:-1] <= tile_depths[1:]))
