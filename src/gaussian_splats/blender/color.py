from __future__ import annotations

import torch


def linear_to_srgb(color: torch.Tensor) -> torch.Tensor:
    return torch.where(
        color <= 0.0031308,
        color * 12.92,
        1.055 * color.clamp_min(0).pow(1 / 2.4) - 0.055,
    )


def srgb_to_linear(color: torch.Tensor) -> torch.Tensor:
    return torch.where(
        color <= 0.04045,
        color / 12.92,
        ((color.clamp_min(0) + 0.055) / 1.055).pow(2.4),
    )
