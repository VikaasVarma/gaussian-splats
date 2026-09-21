from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Self

import torch
import torch.nn.functional as F

from .utils import quaternion_to_rotation_matrix


@dataclass(kw_only=True)
class Camera(ABC):
    rotation: torch.Tensor = field(default_factory=lambda: torch.tensor([1.0, 0.0, 0.0, 0.0]))
    translation: torch.Tensor = field(default_factory=lambda: torch.zeros(3))
    image_size: tuple[int, int] = (640, 480)

    @abstractmethod
    def project_gaussian(
        self, mean: torch.Tensor, rotation: torch.Tensor, scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def fit_to_points(self, points: torch.Tensor) -> Self: ...

    @property
    def rotation_matrix(self) -> torch.Tensor:
        return quaternion_to_rotation_matrix(F.normalize(self.rotation, dim=-1))

    @property
    def position(self) -> torch.Tensor:
        return -(self.rotation_matrix.T @ self.translation)

    def to(self, device: torch.device | str) -> Self:
        self.rotation = self.rotation.to(device)
        self.translation = self.translation.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return self.translation.device

    @property
    def dtype(self) -> torch.dtype:
        return self.translation.dtype


@dataclass(kw_only=True)
class PinholeCamera(Camera):
    """OpenCV pinhole camera with a world-to-camera quaternion."""

    focal_length: tuple[float, float] = (500.0, 500.0)
    principal_point: tuple[float, float] = (320.0, 240.0)
    intrinsics: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        fx, fy = self.focal_length
        cx, cy = self.principal_point
        self.intrinsics = torch.tensor(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ],
            dtype=self.dtype,
            device=self.device,
        )

    def to(self, device: torch.device | str) -> Self:
        super().to(device)
        self.intrinsics = self.intrinsics.to(device)
        return self

    def project_gaussian(
        self,
        mean: torch.Tensor,  # N x 3
        rotation: torch.Tensor,  # N x 3 x 3
        scale: torch.Tensor,  # N x 3
    ) -> tuple[
        torch.Tensor,  # Projected Means: N x 3
        torch.Tensor,  # 2D covariance: N x 2 x 2
    ]:
        camera_rotation = self.rotation_matrix

        # Transform gaussian means to pixel space
        mean = mean @ camera_rotation.T + self.translation
        x, y, z = mean.unbind(dim=-1)
        z = torch.where(z.abs() >= 1e-4, z, torch.sign(z) * 1e-4)
        fx, fy = self.focal_length
        width, height = self.image_size

        mean = mean @ self.intrinsics.T
        mean = mean[:, :2] / mean[:, 2:]
        mean = torch.cat((mean, z[:, None]), dim=-1)

        # Linearize the perspective projection around each Gaussian mean.
        tan_half_fov_x = width / (2 * fx)
        tan_half_fov_y = height / (2 * fy)
        zero = torch.zeros_like(z)
        jacobian = torch.stack(
            (
                fx / z,
                zero,
                -fx * (x / z).clamp(-1.3 * tan_half_fov_x, 1.3 * tan_half_fov_x) / z,
                zero,
                fy / z,
                -fy * (y / z).clamp(-1.3 * tan_half_fov_y, 1.3 * tan_half_fov_y) / z,
            ),
            dim=-1,
        ).view(-1, 2, 3)

        covariance_factor = jacobian @ camera_rotation @ rotation * scale[:, None]
        covariance = covariance_factor @ covariance_factor.mT
        covariance.diagonal(dim1=-2, dim2=-1).add_(0.3)  # Inverse covariance stability
        return mean, covariance

    @torch.no_grad()
    def fit_to_points(self, points: torch.Tensor) -> Self:
        self.to(points.device)
        if len(points) > 10_000:
            points = points[torch.randint(len(points), (10_000,), device=points.device)]

        lower, upper = torch.aminmax(points, dim=0)
        center = 0.5 * (lower + upper)
        radius = 0.5 * (upper - lower).norm().clamp_min(1e-3)

        fx, fy = self.focal_length
        width, height = self.image_size
        tan_half_fov = min(width / (2 * fx), height / (2 * fy))
        distance = 1.1 * radius / math.sin(math.atan(tan_half_fov))

        rotation = self.rotation_matrix
        position = center - distance * rotation[2]
        self.translation.copy_(-(rotation @ position))
        return self
