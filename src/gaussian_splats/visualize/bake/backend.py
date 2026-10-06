from dataclasses import dataclass, field
from pathlib import Path

import torch
from starlette.responses import JSONResponse

from gaussian_splats.bake.query import query_scene
from gaussian_splats.bake.renderers.cycles import CyclesBackend
from gaussian_splats.bake.renderers.phong import PhongBackend
from gaussian_splats.bake.renderers.workbench import WorkbenchBackend
from gaussian_splats.bake.sample import sample_splats
from gaussian_splats.splats.splats import GaussianSplat
from gaussian_splats.visualize import Frame
from gaussian_splats.visualize.gaussian.backend import render_gaussians
from gaussian_splats.visualize.scene.backend import RAY_RENDERERS, SceneViewer
from gaussian_splats.visualize.server import Page, upload


@dataclass
class BakeViewer:
    scene: SceneViewer = field(default_factory=SceneViewer)
    splats: GaussianSplat | None = None

    @property
    def controller(self):
        return self.scene.controller

    def load_scene(self, path, filename):
        self.splats = None
        self.scene.load_scene(path, filename)

    def backend(self, settings):
        renderer = settings.get("renderer", "torch-workbench")
        match renderer:
            case "cycles":
                return CyclesBackend(
                    self.scene.session,
                    samples=int(settings.get("samples", 1)),
                    batch_size=int(settings.get("ray_batch_size", 4096)),
                    exposure=float(settings.get("exposure", 4.0)),
                )
            case "torch-workbench":
                return WorkbenchBackend(
                    ambient_strength=float(settings.get("ambient_strength", 0.05)),
                    diffuse_strength=float(settings.get("diffuse_strength", 1.0)),
                )
            case "phong":
                return PhongBackend(
                    ambient_strength=float(settings.get("ambient_strength", 0.25)),
                    diffuse_strength=float(settings.get("diffuse_strength", 1.0)),
                    specular_strength=float(settings.get("specular_strength", 1.0)),
                )
            case _:
                raise ValueError(f"Unsupported ray renderer: {renderer}")

    def render(self, settings, width, height):
        if self.controller is None:
            return None
        camera = self.controller.camera
        available = max(2, width - 1)
        left_width, right_width = available // 2, available - available // 2
        left = self.backend(settings).render(
            self.scene.scene,
            camera,
            left_width,
            height,
            exposure=float(settings.get("exposure", 4.0)),
        )
        if self.splats is None:
            right = Frame(
                left.new_zeros((height, right_width, 3)), metadata={"X-Total-Splats": "0"}
            )
        else:
            right = render_gaussians(self.splats, camera, settings, right_width, height)
        divider = right.image.new_ones((height, 1, 3))
        return Frame(
            torch.cat((left.to(right.image.device), divider, right.image), dim=1),
            right.timings,
            right.metadata,
        )

    def bake(self, n_splats, settings):
        scene = self.scene.scene
        if scene is None:
            raise ValueError("Load a scene before baking")
        sampled, triangle_id, barycentric = sample_splats(
            scene,
            n_splats=n_splats,
            sigma=float(settings.get("sigma", 0.65)),
        )
        baked, _ = query_scene(
            scene, sampled, self.backend(settings), triangle_id, barycentric
        )
        self.splats = baked.to(scene.vertices.device).eval()
        return {"splats": self.splats.num_points}

    def close(self):
        self.splats = None
        self.scene.close()


def page():
    page = Page(BakeViewer, Path(__file__).parent)

    async def scene(request):
        return await upload(
            request, page, page.backend.load_scene, "X-Scene-Name", {".gltf", ".glb"}
        )

    async def bake(request):
        data = await request.json()
        result = await page.run(
            page.backend.bake,
            max(1, int(data.get("n_splats", 10_000))),
            data,
        )
        return JSONResponse(result)

    async def renderers(request):
        return JSONResponse(RAY_RENDERERS)

    page.extra_routes = [
        ("/scene", scene, ["POST"]),
        ("/bake", bake, ["POST"]),
        ("/renderers", renderers, ["GET"]),
    ]
    return page
