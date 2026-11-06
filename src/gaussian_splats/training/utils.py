import math
from collections.abc import Sequence
from functools import lru_cache

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torchmetrics.functional.image import (
    structural_similarity_index_measure,
)

from ..splats.camera import Camera
from ..splats.splats import GaussianSplat
from ..splats.utils import quaternion_to_rotation_matrix, rgb_to_sh


def initialize_splats(
    views: Sequence[tuple[Camera, Tensor]], count: int, sh_degree: int
) -> GaussianSplat:
    positions = torch.stack([camera.position.cpu() for camera, _ in views])
    lower, upper = positions.amin(0), positions.amax(0)
    center, extent = (lower + upper) / 2, (upper - lower).clamp_min(1e-3)
    splats = GaussianSplat(count, sh_degree=sh_degree)
    spacing = (torch.prod(3 * extent) / count).pow(1 / 3) * 0.5
    with torch.no_grad():
        splats.mean.copy_(center + (torch.rand_like(splats.mean) - 0.5) * 3 * extent)
        splats.rotation.zero_()
        splats.rotation[:, 0] = 1
        splats.scale.fill_(math.log(spacing.item()))
        splats.opacity.fill_(torch.logit(torch.tensor(0.1)).item())
        splats.color.zero_()
        splats.color[:, 0] = rgb_to_sh(torch.rand(count, 3))
        splats.normals.zero_()
    return splats


@torch.no_grad()
def noise_splats(splats: GaussianSplat, noise_lr: float, position_lr: float) -> None:
    # Perturb positions scaled by opacity and learning rate
    rotation = quaternion_to_rotation_matrix(F.normalize(splats.rotation, dim=-1))
    scale = splats.scale.exp()
    factor = rotation * scale[:, None, :]
    covariance = factor @ factor.transpose(-1, -2)
    gate = torch.sigmoid(100 * (1 - splats.opacity.sigmoid() - 0.995))

    noise = torch.randn_like(splats.mean) * gate * noise_lr * position_lr
    noise = (covariance @ noise[..., None]).squeeze(-1)

    splats.mean.add_(noise)


def ssim(image: Tensor, target: Tensor) -> Tensor:
    image, target = (rearrange(value, "h w c -> 1 c h w") for value in (image, target))
    return structural_similarity_index_measure(image, target, data_range=1.0)


@torch.no_grad()
def position_noise(splats: GaussianSplat, noise_lr: float, position_lr: float) -> Tensor:
    # Perturb positions scaled by opacity and learning rate
    rotation = quaternion_to_rotation_matrix(F.normalize(splats.rotation, dim=-1))
    scale = splats.scale.exp()
    factor = rotation * scale[:, None, :]
    covariance = factor @ factor.transpose(-1, -2)
    gate = torch.sigmoid(100 * (1 - splats.opacity.sigmoid() - 0.995))
    noise = torch.randn_like(splats.mean) * gate * noise_lr * position_lr
    return (covariance @ noise[..., None]).squeeze(-1)


@lru_cache
def binomial_coefficients(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    degree = torch.arange(1, 52, device=device)
    coefficients = torch.tensor(
        [[math.comb(n, k) if k <= n else 0 for k in range(1, 52)] for n in range(52)],
        device=device,
        dtype=dtype,
    )
    coefficients *= (-1.0) ** (degree - 1) / degree.sqrt()
    return coefficients, degree


def normalize_copies(opacity: Tensor, scale: Tensor, copies: Tensor) -> tuple[Tensor, Tensor]:
    copies = copies.clamp(1, 51)
    new_opacity = 1 - (1 - opacity).pow(1 / copies)
    coefficients, degree = binomial_coefficients(opacity.device, opacity.dtype)
    terms = coefficients[copies] * new_opacity[:, None].pow(degree)
    terms *= degree <= copies[:, None]
    new_scale = scale * (opacity / terms.sum(-1)).unsqueeze(-1)
    return new_opacity.clamp(0.005, 1 - 1e-6), new_scale


def sample_counts(weights: Tensor, count: int) -> Tensor:
    # Sample weighted by opacity
    indices = torch.multinomial(weights, count, replacement=True)
    return torch.bincount(indices, minlength=len(weights))


def append_empty_splats(splats: GaussianSplat, count: int) -> GaussianSplat:
    extended = GaussianSplat(0, sh_degree=splats.sh_degree).to(splats.mean)
    for name in ("mean", "rotation", "scale", "opacity", "color", "normals"):
        value = getattr(splats, name)
        value = torch.cat((value.detach(), value.new_zeros(count, *value.shape[1:])))
        setattr(extended, name, torch.nn.Parameter(value))
    return extended


@torch.no_grad()
def copy_splats(splats: GaussianSplat, counts: Tensor, output_indices: Tensor) -> Tensor:
    parents = counts.nonzero().squeeze(-1)
    copies = counts[counts > 0]

    source_indices = parents.repeat_interleave(copies)
    opacity = splats.opacity.sigmoid().squeeze(-1)[parents]
    scale = splats.scale[parents].exp()
    new_opacity, new_scale = normalize_copies(opacity, scale, copies + 1)
    output_opacity = new_opacity.repeat_interleave(copies)
    output_scale = new_scale.repeat_interleave(copies, dim=0)

    # Copy parameters
    for name in ("mean", "rotation", "color", "normals"):
        value = getattr(splats, name)
        value.index_copy_(0, output_indices, value[source_indices])

    # Insert normalized opacity and scale
    splats.opacity.index_copy_(0, parents, torch.logit(new_opacity)[:, None])
    splats.opacity.index_copy_(0, output_indices, torch.logit(output_opacity)[:, None])
    splats.scale.index_copy_(0, parents, new_scale.log())
    splats.scale.index_copy_(0, output_indices, output_scale.log())
    return torch.cat((parents, output_indices)).unique()


@torch.no_grad()
def relocate_splats(
    splats: GaussianSplat, optimizer: torch.optim.Optimizer, max_splats: int
) -> None:
    reset_indices = []

    # Replace nearly transparent splats with copies of visible ones
    opacity = splats.opacity.sigmoid().squeeze(-1)
    dead_mask = opacity < 0.005
    dead_indices = dead_mask.nonzero().squeeze(-1)
    if 0 < len(dead_indices) < len(opacity):
        reset_indices.append(
            copy_splats(
                splats=splats,
                counts=sample_counts(opacity.masked_fill(dead_mask, 0), len(dead_indices)),
                output_indices=dead_indices,
            )
        )

    # Densify if needed
    target_splats = min(max_splats, int(1.05 * len(splats.mean)))
    diff = max(0, target_splats - len(splats.mean))
    if diff:
        opacity = splats.opacity.sigmoid().squeeze(-1)
        extended = append_empty_splats(splats, diff)
        copied_indices = copy_splats(
            splats=extended,
            counts=sample_counts(opacity, diff),
            output_indices=torch.arange(len(splats.mean), target_splats, device=opacity.device),
        )
        reset_indices.append(copied_indices[copied_indices < len(splats.mean)])

    # Reset Adam state
    if reset_indices:
        indices = torch.cat(reset_indices).unique()
        for state in optimizer.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key][indices] = 0

    if diff:
        groups = {group["name"]: group for group in optimizer.param_groups}
        for name in ("mean", "rotation", "scale", "opacity", "color", "normals"):
            old, new = getattr(splats, name), getattr(extended, name)
            group = groups.get(name)
            state = optimizer.state.pop(old, None) if group else None
            if state is not None:
                for key in ("exp_avg", "exp_avg_sq"):
                    state[key] = torch.cat((state[key], torch.zeros_like(new[-diff:])))
                optimizer.state[new] = state
                group["params"][0] = new
            setattr(splats, name, new)
