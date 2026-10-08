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
    {
        "value": "cycles",
        "label": "Cycles",
        "settings": (
            {
                "name": "samples",
                "label": "Samples",
                "value": 1,
                "min": 1,
                "step": 1,
            },
            {
                "name": "exposure",
                "label": "Exposure",
                "value": 4.0,
                "min": -10.0,
                "max": 10.0,
                "step": 0.1,
            },
        ),
    },
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

RAY_RENDERERS = tuple(
    {
        **renderer,
        "settings": tuple(
            {
                **setting,
                "value": {"samples": 32, "exposure": 0}.get(setting["name"], setting["value"]),
            }
            if renderer["value"] == "cycles"
            else setting
            for setting in renderer.get("settings", ())
        )
        + (
            {
                "name": "sigma",
                "label": "Sigma",
                "value": 0.6825 if renderer["value"] == "cycles" else 0.65,
                "min": 0,
                "step": "any",
            },
            {
                "name": "sh_degree",
                "label": "SH degree",
                "value": 6 if renderer["value"] == "cycles" else 2,
                "min": 0,
                "max": 3 if renderer["value"] == "torch-workbench" else 12,
                "step": 1,
            },
            {
                "name": "view_samples",
                "label": "Directions per splat",
                "value": 128 if renderer["value"] == "cycles" else 32,
                "min": 1,
                "step": 1,
            },
        )
        + (
            (
                {
                    "name": "cycles_device",
                    "label": "Cycles query device",
                    "value": "CPU",
                    "options": ["CPU", "OPTIX"],
                },
                {
                    "name": "cycles_workers",
                    "label": "Cycles workers",
                    "value": 1,
                    "min": 1,
                    "step": 1,
                },
                {
                    "name": "ray_batch_size",
                    "label": "Query batch size",
                    "value": 1048576,
                    "min": 1,
                    "step": 1,
                },
                {
                    "name": "eps",
                    "label": "Ray offset (eps)",
                    "value": 1e-4,
                    "min": 0,
                    "step": "any",
                },
            )
            if renderer["value"] == "cycles"
            else ()
        ),
    }
    for renderer in RENDERERS
    if renderer["value"] in {"cycles", "torch-workbench", "phong"}
)


@dataclass
class SceneViewer:
    session: BlenderSession | None = None
    controller: CameraController | None = None
    scene: Scene | None = None

    def load_scene(self, path, filename):
        self.close()
        self.session = BlenderSession(path)
        self.session.__enter__()
        self.scene = self.session.scene.to(default_device())
        self.controller = CameraController(fit_camera(self.scene))

    def render(self, settings, width, height):
        if self.controller is None:
            return None
        renderer = settings.get("renderer", "cycles")
        camera = self.controller.camera
        match renderer:
            case "cycles" | "eevee" | "workbench":
                image = self.session.render(
                    camera,
                    width,
                    height,
                    renderer=renderer,
                    samples=int(settings.get("samples", 1)),
                    exposure=float(settings.get("exposure", 0.0)),
                )
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
    page = Page(SceneViewer, Path(__file__).parent)

    async def scene(request):
        return await upload(
            request, page, page.backend.load_scene, "X-Scene-Name", {".gltf", ".glb"}
        )

    async def renderers(request):
        return JSONResponse(RENDERERS)

    page.extra_routes = [
        ("/scene", scene, ["POST"]),
        ("/renderers", renderers, ["GET"]),
    ]
    return page
