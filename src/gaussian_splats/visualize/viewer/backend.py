import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from starlette.responses import JSONResponse, Response

from gaussian_splats.splats.camera import PinholeCamera
from gaussian_splats.splats.splats import GaussianSplat
from gaussian_splats.visualize import default_device
from gaussian_splats.visualize.camera import CameraController, _quaternion_from_rotation
from gaussian_splats.visualize.gaussian.backend import GaussianViewer
from gaussian_splats.visualize.server import Page, upload


def _ply(splats, settings):
    count = min(max(1, int(settings.get("num_splats") or splats.num_points)), splats.num_points)
    indices = (
        torch.linspace(0, splats.num_points - 1, count, device=splats.mean.device).round().long()
    )
    requested_degree = settings.get("sh_degree")
    requested_degree = (
        int(requested_degree) if requested_degree not in (None, "") else splats.sh_degree
    )
    degree = min(3, max(0, requested_degree), splats.sh_degree)
    color = splats.color[indices, : (degree + 1) ** 2].detach().cpu().numpy()
    rest = color[:, 1:].copy()

    # Playcanvas expects SH from camera -> point (negate odd coefficients)
    for band in range(1, degree + 1, 2):
        rest[:, band * band - 1 : (band + 1) * (band + 1) - 1] *= -1
    mean = splats.mean[indices].detach().cpu().numpy()
    normals = np.zeros_like(mean)
    rotation = splats.rotation[indices].detach().cpu().numpy()
    scale = splats.scale[indices].detach().cpu().numpy()
    opacity = splats.opacity[indices].detach().cpu().numpy()

    scale_factor = max(float(settings.get("scale", 1.0)), 1e-6)
    scale += np.log(scale_factor)
    opacity = torch.sigmoid(torch.from_numpy(opacity))
    opacity_factor = max(float(settings.get("opacity", 1.0)), 0.0)
    opacity = (opacity * opacity_factor).clamp(max=1 - 1e-6)
    opacity = torch.logit(opacity).numpy()

    fields = (
        ["x", "y", "z", "nx", "ny", "nz"]
        + [f"f_dc_{i}" for i in range(3)]
        + [f"f_rest_{i}" for i in range((color.shape[1] - 1) * 3)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    )
    values = np.concatenate(
        (
            mean,
            normals,
            color[:, 0],
            rest.transpose(0, 2, 1).reshape(count, -1),
            opacity,
            scale,
            rotation,
        ),
        axis=1,
    )
    vertex = np.empty(count, dtype=[(field, "f4") for field in fields])
    for index, field in enumerate(fields):
        vertex[field] = values[:, index]

    output = BytesIO()
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(output)
    return output.getvalue(), count


@dataclass
class ViewerBackend(GaussianViewer):
    def _view_camera(self, view):
        points = self.splats.mean.detach()
        lower, upper = torch.aminmax(points, dim=0)
        center = (lower + upper) * 0.5
        radius = 0.5 * torch.linalg.vector_norm(upper - lower)
        distance = 1.1 * radius / math.sin(math.atan(512 / 1000))
        base = center + points.new_tensor((0.0, -distance, 0.2 * distance))
        angle = (0.0, math.pi / 6, math.pi, 7 * math.pi / 6)[int(view) % 4]
        cosine, sine = math.cos(angle), math.sin(angle)
        offset = base - center
        position = center + points.new_tensor(
            (
                cosine * offset[0] - sine * offset[1],
                sine * offset[0] + cosine * offset[1],
                offset[2],
            )
        )
        forward = torch.nn.functional.normalize(center - position, dim=0)
        up = points.new_tensor((0.0, 0.0, 1.0))
        right = torch.nn.functional.normalize(torch.linalg.cross(forward, up), dim=0)
        camera_up = torch.linalg.cross(right, forward)
        rotation = torch.stack((right, camera_up, -forward))
        return PinholeCamera(
            rotation=_quaternion_from_rotation(rotation),
            translation=-(rotation @ position),
            image_size=(512, 512),
            focal_length=(500.0, 500.0),
            principal_point=(256.0, 256.0),
        )

    def load_checkpoint(self, path, filename):
        device = self.splats.mean.device if self.splats is not None else default_device()
        self.splats = GaussianSplat.from_checkpoint(path).to(device).eval()
        rotation = self.splats.mean.new_tensor(
            [math.cos(math.pi / 8), -math.sin(math.pi / 8), 0.0, 0.0]
        )
        self.controller = CameraController(
            PinholeCamera(rotation=rotation).fit_to_points(self.splats.mean)
        )
        self.checkpoint_name = filename

    def camera_info(self, view=0):
        if self.controller is None:
            return {
                "position": [0.0, 0.0, 3.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "fov": 60.0,
            }
        self.controller.camera = self._view_camera(view)
        self.controller.__post_init__()
        camera = self.controller.camera
        w, h = camera.image_size
        rotation = torch.nn.functional.normalize(camera.rotation, dim=-1)
        frame = rotation.new_tensor([math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0])
        camera_to_world = rotation * rotation.new_tensor([1.0, -1.0, -1.0, -1.0])
        fw, fx, fy, fz = frame
        cw, cx, cy, cz = camera_to_world
        rotation = torch.stack(
            (
                fw * cw - fx * cx - fy * cy - fz * cz,
                fw * cx + fx * cw + fy * cz - fz * cy,
                fw * cy - fx * cz + fy * cw + fz * cx,
                fw * cz + fx * cy - fy * cx + fz * cw,
            )
        )
        position = camera.position.new_tensor(
            [camera.position[0], camera.position[2], -camera.position[1]]
        )
        return {
            "position": position.detach().cpu().tolist(),
            "rotation": [
                rotation[1].item(),
                rotation[2].item(),
                rotation[3].item(),
                rotation[0].item(),
            ],
            "fov": float(2 * np.degrees(np.arctan(h / (2 * camera.focal_length[1])))),
        }

    def splat_data(self, settings):
        if self.splats is None:
            return b"", 0
        return _ply(self.splats, settings)


def page():
    page = Page(ViewerBackend.create, Path(__file__).parent)

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
                "sh_degree": splats.sh_degree if splats is not None else 0,
            }

        return JSONResponse(await page.run(info))

    async def camera_info(request):
        view = int(request.query_params.get("view", 0))
        return JSONResponse(await page.run(page.backend.camera_info, view))

    async def splats(request):
        settings = dict(request.query_params)
        data, count = await page.run(page.backend.splat_data, settings)
        if not data:
            return Response(status_code=204)
        return Response(
            data,
            media_type="application/octet-stream",
            headers={"Cache-Control": "no-store", "X-Total-Splats": str(count)},
        )

    page.extra_routes = [
        ("/checkpoint", checkpoint, ["POST"]),
        ("/checkpoint", checkpoint_info, ["GET"]),
        ("/camera-info", camera_info, ["GET"]),
        ("/splats.ply", splats, ["GET"]),
    ]
    return page
