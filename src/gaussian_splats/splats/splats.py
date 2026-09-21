from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Self

import torch
import torch.nn.functional as F
from torch import nn

from .utils import rgb_to_sh


@dataclass(frozen=True)
class ProjectedGaussianSplat:
    mean: torch.Tensor  # N x 3
    covariance: torch.Tensor  # N x 2 x 2
    color: torch.Tensor  # N x 3
    opacity: torch.Tensor  # N x 1

    @cached_property
    def inverse_covariance(self) -> torch.Tensor:  # N x 2 x 2
        return torch.linalg.inv(self.covariance)


class GaussianSplat(nn.Module):
    mean: torch.Tensor  # N x 3
    rotation: torch.Tensor  # N x 4
    scale: torch.Tensor  # N x 3
    opacity: torch.Tensor  # N x 1
    color: torch.Tensor  # N x (sh_degree + 1) ** 2 x 3

    def __init__(self, num_points: int, sh_degree: int = 3):
        super().__init__()
        self.sh_degree = sh_degree
        self.mean = nn.Parameter(torch.randn(num_points, 3))
        self.rotation = nn.Parameter(F.normalize(torch.randn(num_points, 4)))
        self.scale = nn.Parameter(torch.randn(num_points, 3))
        self.opacity = nn.Parameter(torch.randn(num_points, 1))
        color = torch.zeros(num_points, (sh_degree + 1) ** 2, 3)
        color[:, 0, :] = rgb_to_sh(torch.rand(num_points, 3))
        self.color = nn.Parameter(color)

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path) -> Self:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        splats = cls(len(state["mean"]), int(state["color"].shape[1] ** 0.5) - 1)
        splats.load_state_dict(state)
        return splats

    @property
    def num_points(self) -> int:
        return len(self.mean)
