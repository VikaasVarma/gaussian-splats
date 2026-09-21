"""Triton tile-intersection emission and Torch global ordering."""

import warnings

import torch
import triton
import triton.language as tl


@triton.jit
def _emit_intersections(
    bounds,
    depths,
    counts,
    offsets,
    keys,
    gaussian_ids,
    tiles_w: tl.constexpr,
):
    gaussian_id = tl.program_id(0)
    count = tl.load(counts + gaussian_id)
    offset = tl.load(offsets + gaussian_id)
    min_x = tl.load(bounds + 4 * gaussian_id)
    max_x = tl.load(bounds + 4 * gaussian_id + 1)
    min_y = tl.load(bounds + 4 * gaussian_id + 2)
    width = max_x - min_x
    depth_bits = tl.cast(tl.load(depths + gaussian_id), tl.int32, bitcast=True).to(tl.int64)

    # Assign Gaussian IDs to tiles.
    local_id = 0
    while local_id < count:
        tile_x = min_x + local_id % width
        tile_y = min_y + local_id // width

        # Get corresponding tile IDs
        tile_id = tile_y * tiles_w + tile_x
        output_id = offset + local_id
        key = (tile_id.to(tl.int64) << 32) | (depth_bits & 0xFFFFFFFF)
        tl.store(keys + output_id, key)
        tl.store(gaussian_ids + output_id, gaussian_id)
        local_id += 1


def emit_intersections(
    bounds: torch.Tensor,
    depths: torch.Tensor,
    counts: torch.Tensor,
    tiles_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Emit packed ``(tile ID, depth)`` keys and their Gaussian IDs."""
    total = int(counts.sum().item())
    offsets = (counts.cumsum(0) - counts).to(torch.int32)
    keys = torch.empty(total, dtype=torch.int64, device=depths.device)
    gaussian_ids = torch.empty(total, dtype=torch.int32, device=depths.device)
    if len(counts):
        if depths.dtype != torch.float32:
            warnings.warn(
                f"Casting depths from {depths.dtype} to float32 for key packing.",
                RuntimeWarning,
                stacklevel=2,
            )
            depths = depths.float()
        if not depths.is_contiguous():
            depths = depths.contiguous()
        _emit_intersections[(len(counts),)](
            bounds,
            depths,
            counts,
            offsets,
            keys,
            gaussian_ids,
            tiles_w,
        )
    return keys, gaussian_ids


def sort_intersections(
    keys: torch.Tensor, gaussian_ids: torch.Tensor, num_tiles: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Globally sort intersections and return each tile's half-open range."""
    # Order by tile ID, then front-to-back depth.
    keys, order = keys.sort()
    gaussian_ids = gaussian_ids[order]
    counts = torch.bincount(keys >> 32, minlength=num_tiles)
    ends = counts.cumsum(0)

    # Convert sorted tile IDs into per-tile ranges.
    return gaussian_ids, torch.stack((ends - counts, ends), dim=-1)
