from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from gaussian_splats.splats.camera import PinholeCamera

from .api import BlenderSession


class BlenderCameraRenderer:
    def __init__(
        self,
        session: BlenderSession,
        renderer: str = "cycles",
        samples: int = 1,
    ) -> None:
        self.session = session
        self.renderer = renderer
        self.samples = samples

    def render(self, camera: PinholeCamera, width: int, height: int) -> torch.Tensor:
        output = self.session.request(
            {
                "command": "render",
                "renderer": self.renderer,
                "samples": self.samples,
                "width": width,
                "height": height,
                "position": camera.position.tolist(),
                "rotation": camera.rotation_matrix.tolist(),
                "focal_length": list(camera.focal_length),
            },
            output_suffix=".png",
            marker="RENDER_RESULT",
        )
        image = np.array(Image.open(output).convert("RGB"), copy=True)
        return torch.from_numpy(image).float() / 255
