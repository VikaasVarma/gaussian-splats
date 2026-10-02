import math
from dataclasses import dataclass, field

import torch

from gaussian_splats.splats.camera import PinholeCamera


def _quaternion_from_rotation(rotation: torch.Tensor) -> torch.Tensor:
    trace = rotation.trace()
    if trace > 0:
        scale = torch.sqrt(trace + 1) * 2
        return rotation.new_tensor(
            [
                scale / 4,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )

    diagonal = rotation.diagonal()
    index = int(diagonal.argmax())
    next_index = (index + 1) % 3
    last_index = (index + 2) % 3
    scale = (
        torch.sqrt(
            torch.clamp(1 + diagonal[index] - diagonal[next_index] - diagonal[last_index], min=1e-8)
        )
        * 2
    )
    quaternion = rotation.new_zeros(4)
    quaternion[index + 1] = scale / 4
    quaternion[0] = (rotation[last_index, next_index] - rotation[next_index, last_index]) / scale
    quaternion[next_index + 1] = (rotation[next_index, index] + rotation[index, next_index]) / scale
    quaternion[last_index + 1] = (rotation[last_index, index] + rotation[index, last_index]) / scale
    return quaternion


@dataclass
class CameraController:
    camera: PinholeCamera
    world_up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    speed: float = 2.0
    sensitivity: float = 0.002
    yaw: float = 0.0
    pitch: float = 0.0
    position: torch.Tensor = field(init=False)
    forward_axis: torch.Tensor = field(init=False)
    up_axis: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.position = self.camera.position
        self.up_axis = self.position.new_tensor(self.world_up)
        forward = self.camera.rotation_matrix[2]
        self.pitch = math.asin(float(forward.dot(self.up_axis).clamp(-1, 1)))
        self.forward_axis = forward - forward.dot(self.up_axis) * self.up_axis
        self.forward_axis = self.forward_axis / self.forward_axis.norm().clamp_min(1e-8)

    def update(self, move: list[float], look: list[float], dt: float) -> None:
        self.yaw += look[0] * self.sensitivity
        self.pitch = max(-1.57, min(1.57, self.pitch + look[1] * self.sensitivity))

        sin_yaw, cos_yaw = math.sin(self.yaw), math.cos(self.yaw)
        base_right = torch.linalg.cross(self.up_axis, self.forward_axis)
        forward = cos_yaw * self.forward_axis + sin_yaw * base_right
        right = torch.linalg.cross(self.up_axis, forward)
        up = self.up_axis
        direction = move[0] * right + move[1] * up + move[2] * forward
        self.position += self.speed * dt * direction / direction.norm().clamp_min(1)

        pitch_cos, pitch_sin = math.cos(self.pitch), math.sin(self.pitch)
        camera_forward = pitch_cos * forward + pitch_sin * up
        camera_up = torch.linalg.cross(camera_forward, right)
        rotation = torch.stack((right, camera_up, camera_forward))
        self.camera.rotation = _quaternion_from_rotation(rotation)
        self.camera.translation = -(self.camera.rotation_matrix @ self.position)
