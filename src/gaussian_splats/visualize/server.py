"""Serve rasterizer frames to the browser viewer."""

import json
import os
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import typer
import uvicorn
from PIL import Image
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from gaussian_splats.splats.camera import PinholeCamera
from gaussian_splats.splats.rasterize import rasterize
from gaussian_splats.splats.splats import GaussianSplat
from gaussian_splats.timing import record_timings

from .controls import CameraController

CHECKPOINT = "GAUSSIAN_SPLATS_CHECKPOINT"


def render_frame(
    splats: GaussianSplat, camera: PinholeCamera, **options: int | float | str
) -> tuple[bytes, dict[str, dict[str, float]]]:
    with record_timings(splats.mean.device) as times:
        image = rasterize(splats, camera, **options)

    output = BytesIO()
    Image.fromarray(image, "RGB").save(output, "JPEG", quality=75)
    return output.getvalue(), times


def frame(request: Request) -> Response:
    camera = request.app.state.camera
    params = request.query_params

    width = int(params.get("width", camera.image_size[0]))
    height = int(params.get("height", camera.image_size[1]))
    camera = replace(
        camera,
        image_size=(width, height),
        principal_point=(width / 2, height / 2),
    )

    options = {
        "tile_size": int(params.get("tile_size", 16)),
        "opacity_threshold": float(params.get("opacity_threshold", 0.999)),
        "near": float(params.get("near", 0.1)),
        "far": float(params.get("far", 100)),
        "num_splats": int(params.get("num_splats", 100)),
        "rendering_mode": params.get("rendering_mode", "gaussian"),
        "confidence": float(params.get("confidence", 0.95)),
    }

    image, timings = render_frame(request.app.state.splats, camera, **options)
    return Response(
        image,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store",
            "X-Timings": json.dumps(timings),
            "X-Total-Splats": str(request.app.state.splats.num_points),
        },
    )


def ping(request: Request) -> Response:
    return Response(status_code=204)


async def control(request: Request) -> Response:
    data = await request.json()
    request.app.state.controller.update(data["move"], data["look"], data["dt"])
    return Response(status_code=204)


def create_app() -> Starlette:
    splats = (
        GaussianSplat.from_checkpoint(os.getenv(CHECKPOINT))
        if os.getenv(CHECKPOINT) is not None
        else GaussianSplat(1_000_000)
    )
    camera = PinholeCamera().fit_to_points(splats.mean)

    app = Starlette(
        routes=[
            Route("/frame.jpg", frame),
            Route("/ping", ping),
            Route("/camera", control, methods=["POST"]),
            Mount(
                "/",
                app=StaticFiles(packages=[("gaussian_splats.visualize", "web")], html=True),
            ),
        ]
    )
    app.state.splats = splats.eval()
    app.state.camera = camera
    app.state.controller = CameraController(camera)
    return app


def serve(
    checkpoint: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 7007,
) -> None:
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise typer.BadParameter(f"Checkpoint does not exist: {checkpoint}")
        os.environ[CHECKPOINT] = str(checkpoint)
    else:
        os.environ.pop(CHECKPOINT, None)

    typer.echo(f"Viewer: http://{host}:{port}")
    uvicorn.run(
        "gaussian_splats.visualize.server:create_app",
        factory=True,
        host=host,
        port=port,
        log_level="warning",
        reload=True,
    )


def main() -> None:
    typer.run(serve)


if __name__ == "__main__":
    main()
