import math
import os
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from starlette.responses import JSONResponse

from gaussian_splats.splats.camera import PinholeCamera
from gaussian_splats.splats.rasterize import rasterize
from gaussian_splats.splats.splats import GaussianSplat
from gaussian_splats.timing import record_timings
from gaussian_splats.visualize import Frame, default_device
from gaussian_splats.visualize.camera import CameraController
from gaussian_splats.visualize.server import Page, upload


def render_gaussians(splats, camera, settings, width, height, rasterizer=rasterize):
    if splats.num_points == 0:
        return Frame(
            splats.mean.new_zeros((height, width, 3)),
            metadata={"X-Total-Splats": "0"},
        )
    tile = int(settings.get("tile_size", 8))
    image_size = ((width + tile - 1) // tile * tile, (height + tile - 1) // tile * tile)
    camera = replace(camera, image_size=image_size, principal_point=(width / 2, height / 2))
    count = min(max(1, int(settings.get("num_splats") or 10_000)), splats.num_points)
    options = {
        "backend": settings.get("backend", "torch"),
        "tile_size": tile,
        "opacity_threshold": float(settings.get("opacity_threshold", 0.999)),
        "near": float(settings.get("near", 0.1)),
        "far": float(settings.get("far", 100)),
        "rendering_mode": settings.get("rendering_mode", "gaussian"),
        "confidence": float(settings.get("confidence", 0.95)),
        # Sample across the checkpoint instead of showing only its first rows.
        "indices": torch.linspace(0, splats.num_points - 1, count, device=splats.mean.device)
        .round()
        .long(),
    }
    timing = (
        record_timings(splats.mean.device)
        if os.getenv("GAUSSIAN_SPLATS_PROFILE", "0") != "0"
        else nullcontext({})
    )
    with torch.no_grad(), timing as groups:
        image = rasterizer(splats, camera, **options)
    timings = {
        f"{group}/{name}": value
        for group, entries in groups.items()
        for name, value in entries.items()
    }
    return Frame(image[:height, :width], timings, {"X-Total-Splats": str(splats.num_points)})


@dataclass
class GaussianViewer:
    splats: GaussianSplat | None = None
    controller: CameraController | None = None
    checkpoint_name: str = ""
    compiled_rasterizer: object | None = None

    @classmethod
    def create(cls):
        return cls()

    def load_checkpoint(self, path, filename):
        device = self.splats.mean.device if self.splats is not None else default_device()
        splats = GaussianSplat.from_checkpoint(path).to(device).eval()
        rotation = splats.mean.new_tensor(
            [math.cos(math.pi / 8), -math.sin(math.pi / 8), 0.0, 0.0]
        )
        camera = PinholeCamera(rotation=rotation).fit_to_points(splats.mean)
        self.controller = CameraController(camera)
        self.splats = splats
        self.checkpoint_name = filename

    def render(self, settings, width, height):
        if self.splats is None:
            return None
        if settings.get("compile_render") == "1" and self.compiled_rasterizer is None:
            self.compiled_rasterizer = torch.compile(rasterize, dynamic=True)
        rasterizer = self.compiled_rasterizer or rasterize
        frame = render_gaussians(
            self.splats, self.controller.camera, settings, width, height, rasterizer
        )
        frame.metadata["X-Checkpoint-Name"] = self.checkpoint_name
        return frame


def page():
    page = Page(GaussianViewer.create, Path(__file__).parent)

    async def checkpoint(request):
        return await upload(
            request, page, page.backend.load_checkpoint, "X-Checkpoint-Name", {".pt"}
        )

    async def checkpoint_info(request):
        def info():
            splats = page.backend.splats
            return {
                "name": page.backend.checkpoint_name,
                "splats": splats.num_points if splats is not None else 0,
            }

        return JSONResponse(await page.run(info))

    page.extra_routes = [
        ("/checkpoint", checkpoint, ["POST"]),
        ("/checkpoint", checkpoint_info, ["GET"]),
    ]
    return page
