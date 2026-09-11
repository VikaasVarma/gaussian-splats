import math
from dataclasses import dataclass, field

import torch

from gaussian_splats.splats.camera import PinholeCamera


@dataclass
class CameraController:
    camera: PinholeCamera
    speed: float = 2.0
    sensitivity: float = 0.002
    yaw: float = 0.0
    pitch: float = 0.0
    position: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.position = self.camera.position

    def update(self, move: list[float], look: list[float], dt: float) -> None:
        self.yaw += look[0] * self.sensitivity
        self.pitch = max(-1.57, min(1.57, self.pitch - look[1] * self.sensitivity))

        sin_yaw, cos_yaw = math.sin(self.yaw), math.cos(self.yaw)
        right = self.position.new_tensor((cos_yaw, 0.0, -sin_yaw))
        up = self.position.new_tensor((0.0, -1.0, 0.0))
        forward = self.position.new_tensor((sin_yaw, 0.0, cos_yaw))
        direction = move[0] * right + move[1] * up + move[2] * forward
        self.position += self.speed * dt * direction / direction.norm().clamp_min(1)

        cy, sy = math.cos(self.yaw / 2), math.sin(self.yaw / 2)
        cp, sp = math.cos(self.pitch / 2), math.sin(self.pitch / 2)
        self.camera.rotation = self.position.new_tensor((cy * cp, -cy * sp, -sy * cp, sy * sp))
        self.camera.translation = -(self.camera.rotation_matrix @ self.position)
