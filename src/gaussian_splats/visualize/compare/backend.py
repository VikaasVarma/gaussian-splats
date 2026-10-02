from dataclasses import dataclass, field
from pathlib import Path

import torch
from starlette.responses import JSONResponse

from gaussian_splats.bake.query import query_scene
from gaussian_splats.bake.renderers.workbench import WorkbenchBackend
from gaussian_splats.bake.sample import sample_splats
from gaussian_splats.splats.splats import GaussianSplat
from gaussian_splats.visualize import Frame
from gaussian_splats.visualize.bake.backend import BlenderViewer
from gaussian_splats.visualize.gaussian.backend import render_gaussians
from gaussian_splats.visualize.server import Page, upload


@dataclass
class ComparisonViewer:
    blender: BlenderViewer = field(default_factory=BlenderViewer)
    splats: GaussianSplat | None = None

    @property
    def controller(self):
        return self.blender.controller

    def load_scene(self, path, filename):
        self.splats = None
        self.blender.load_scene(path, filename)

    def render(self, settings, width, height):
        if self.controller is None:
            return None
        camera = self.controller.camera
        left_width, right_width = max(1, width // 2), max(1, width - width // 2)
        left = WorkbenchBackend().render(self.blender.scene, camera, left_width, height)
        if self.splats is None:
            right = Frame(
                left.new_zeros((height, right_width, 3)), metadata={"X-Total-Splats": "0"}
            )
        else:
            right = render_gaussians(self.splats, camera, settings, right_width, height)
        return Frame(torch.cat((left, right.image), dim=1), right.timings, right.metadata)

    def bake(self, n_splats):
        scene = self.blender.scene
        if scene is None:
            raise ValueError("Load a scene before baking")
        sampled, triangle_id, barycentric = sample_splats(scene, n_splats=n_splats)
        baked, _ = query_scene(scene, sampled, WorkbenchBackend(), triangle_id, barycentric)
        self.splats = baked.to(scene.vertices.device).eval()
        return {"splats": self.splats.num_points}

    def close(self):
        self.splats = None
        self.blender.close()


def page():
    page = Page(ComparisonViewer, Path(__file__).parent)

    async def scene(request):
        return await upload(
            request, page, page.backend.load_scene, "X-Scene-Name", {".gltf", ".glb"}
        )

    async def bake(request):
        data = await request.json()
        result = await page.run(page.backend.bake, max(1, int(data.get("n_splats", 10_000))))
        return JSONResponse(result)

    async def clear(request):
        await page.run(page.backend.close)
        return JSONResponse({"ok": True})

    page.extra_routes = [
        ("/scene", scene, ["POST"]),
        ("/bake", bake, ["POST"]),
        ("/scene/clear", clear, ["POST"]),
    ]
    return page
