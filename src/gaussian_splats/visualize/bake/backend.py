from dataclasses import dataclass
from pathlib import Path

from starlette.responses import JSONResponse

from gaussian_splats.bake.blender import (
    BlenderSession,
    fit_camera,
)
from gaussian_splats.bake.renderers.phong import PhongBackend
from gaussian_splats.bake.renderers.workbench import WorkbenchBackend
from gaussian_splats.bake.scene import Scene
from gaussian_splats.visualize import Frame, default_device
from gaussian_splats.visualize.camera import CameraController
from gaussian_splats.visualize.server import Page, upload

RENDERERS = (
    {"value": "cycles", "label": "Cycles"},
    {"value": "eevee", "label": "Eevee"},
    {"value": "workbench", "label": "Workbench (Blender)"},
    {
        "value": "torch-workbench",
        "label": "Workbench (Torch)",
        "settings": (
            {
                "name": "ambient_strength",
                "label": "Ambient",
                "value": 0.05,
                "min": 0,
                "step": 0.01,
            },
            {
                "name": "diffuse_strength",
                "label": "Diffuse",
                "value": 1.0,
                "min": 0,
                "step": 0.01,
            },
        ),
    },
    {
        "value": "phong",
        "label": "Phong",
        "settings": (
            {
                "name": "ambient_strength",
                "label": "Ambient",
                "value": 0.25,
                "min": 0,
                "step": 0.01,
            },
            {
                "name": "diffuse_strength",
                "label": "Diffuse",
                "value": 1.0,
                "min": 0,
                "step": 0.01,
            },
            {
                "name": "specular_strength",
                "label": "Specular",
                "value": 1.0,
                "min": 0,
                "step": 0.01,
            },
        ),
    },
)


@dataclass
class BlenderViewer:
    session: BlenderSession | None = None
    controller: CameraController | None = None
    scene: Scene | None = None

    def load_scene(self, path, filename):
        self.close()
        self.session = BlenderSession(path)
        self.session.__enter__()
        self.scene = self.session.scene.to(default_device())
        self.controller = CameraController(fit_camera(self.scene), world_up=(0.0, 0.0, 1.0))

    def render(self, settings, width, height):
        if self.controller is None:
            return None
        renderer = settings.get("renderer", "cycles")
        camera = self.controller.camera
        match renderer:
            case "cycles" | "eevee" | "workbench":
                image = self.session.render(camera, width, height, renderer=renderer)
            case "torch-workbench":
                image = WorkbenchBackend(
                    ambient_strength=float(settings.get("ambient_strength", 0.05)),
                    diffuse_strength=float(settings.get("diffuse_strength", 1.0)),
                ).render(self.scene, camera, width, height)
            case "phong":
                image = PhongBackend(
                    ambient_strength=float(settings.get("ambient_strength", 0.25)),
                    diffuse_strength=float(settings.get("diffuse_strength", 1.0)),
                    specular_strength=float(settings.get("specular_strength", 1.0)),
                ).render(self.scene, camera, width, height)
            case _:
                raise ValueError(f"Unsupported renderer: {renderer}")
        return Frame(image)

    def close(self):
        if self.session is not None:
            self.session.__exit__(None, None, None)
        self.session = self.scene = self.controller = None


def page():
    page = Page(BlenderViewer, Path(__file__).parent)

    async def scene(request):
        return await upload(
            request, page, page.backend.load_scene, "X-Scene-Name", {".gltf", ".glb"}
        )

    async def renderers(request):
        return JSONResponse(RENDERERS)

    async def clear(request):
        await page.run(page.backend.close)
        return JSONResponse({"ok": True})

    page.extra_routes = [
        ("/scene", scene, ["POST"]),
        ("/renderers", renderers, ["GET"]),
        ("/scene/clear", clear, ["POST"]),
    ]
    return page
