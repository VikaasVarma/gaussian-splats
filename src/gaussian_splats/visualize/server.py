"""Serve viewer frames over HTTP."""

import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import typer
import uvicorn
from PIL import Image
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .blender import BlenderViewer
from .gaussian import CHECKPOINT_ENV, GaussianViewer


def encode_jpeg(image: Any) -> bytes:
    output = BytesIO()
    Image.fromarray((255 * image.clamp(0, 1)).byte().cpu().numpy(), "RGB").save(
        output, format="JPEG", quality=75
    )
    return output.getvalue()


async def frame(request: Request) -> Response:
    viewer = request.app.state.viewer

    def render() -> Any:
        with request.app.state.render_lock:
            viewer.update_controls(request.app.state.pending_controls)
            request.app.state.pending_controls.clear()
            return viewer.render(request.query_params)

    try:
        result = await asyncio.wrap_future(request.app.state.render_executor.submit(render))
    except Exception as error:
        return JSONResponse({"error": str(error)}, status_code=500)
    if result is None:
        return Response(status_code=204, headers={"X-No-Scene": "1"})
    headers = {
        "Cache-Control": "no-store",
        "X-Timings": json.dumps(result.timings),
        **result.metadata,
    }
    return Response(encode_jpeg(result.image), media_type="image/jpeg", headers=headers)


def ping(request: Request) -> Response:
    return Response(status_code=204)


async def control(request: Request) -> Response:
    data = await request.json()
    with request.app.state.render_lock:
        request.app.state.pending_controls.append(data)
    return Response(status_code=204)


async def blender_frame(request: Request) -> Response:
    viewer = request.app.state.blender_viewer
    if viewer is None:
        return JSONResponse({"error": request.app.state.blender_error}, status_code=503)

    def render() -> Any:
        with request.app.state.render_lock:
            viewer.update_controls(request.app.state.blender_controls)
            request.app.state.blender_controls.clear()
            return viewer.render(request.query_params)

    try:
        result = await asyncio.wrap_future(request.app.state.render_executor.submit(render))
    except Exception as error:
        return JSONResponse({"error": str(error)}, status_code=500)
    if result is None:
        return Response(status_code=204)
    return Response(
        encode_jpeg(result), media_type="image/jpeg", headers={"Cache-Control": "no-store"}
    )


async def blender_control(request: Request) -> Response:
    data = await request.json()
    with request.app.state.render_lock:
        request.app.state.blender_controls.append(data)
    return Response(status_code=204)


async def blender_renderers(request: Request) -> Response:
    viewer = request.app.state.blender_viewer
    if viewer is None:
        return JSONResponse({"error": request.app.state.blender_error}, status_code=503)
    return JSONResponse(viewer.renderers)


async def blender_scene(request: Request) -> Response:
    viewer = request.app.state.blender_viewer
    if viewer is None:
        return JSONResponse({"error": request.app.state.blender_error}, status_code=503)
    name = request.headers.get("X-Scene-Name", "scene.gltf")
    suffix = Path(name).suffix.lower()
    if suffix not in {".gltf", ".glb"}:
        return JSONResponse({"error": "Only .gltf and .glb scenes are supported."}, status_code=415)
    with NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
        path = Path(temporary.name)
        try:
            async for chunk in request.stream():
                temporary.write(chunk)
            temporary.close()
            with request.app.state.render_lock:
                future = request.app.state.render_executor.submit(viewer.load_scene, path)
                await asyncio.wrap_future(future)
        except Exception as error:
            return JSONResponse({"error": str(error)}, status_code=400)
        finally:
            path.unlink(missing_ok=True)
    return Response(status_code=204)


def blender_page(request: Request) -> FileResponse:
    return FileResponse(Path(__file__).parent / "web" / "blender.html")


async def checkpoint(request: Request) -> Response:
    filename = request.headers.get("X-Checkpoint-Name", "checkpoint.ply")
    if Path(filename).suffix.lower() != ".ply":
        return JSONResponse({"error": "Only .ply checkpoints are supported."}, status_code=415)

    with NamedTemporaryFile(suffix=".ply", delete=False) as temporary:
        path = Path(temporary.name)
        try:
            async for chunk in request.stream():
                temporary.write(chunk)
            temporary.close()
            with request.app.state.render_lock:
                future = request.app.state.render_executor.submit(
                    request.app.state.viewer.load_checkpoint, path
                )
                await asyncio.wrap_future(future)
                request.app.state.viewer.checkpoint_name = Path(filename).name
        except Exception as error:
            return JSONResponse({"error": str(error)}, status_code=400)
        finally:
            path.unlink(missing_ok=True)
    return Response(status_code=204)


def create_app() -> Starlette:
    app = Starlette(
        routes=[
            Route("/frame.jpg", frame),
            Route("/ping", ping),
            Route("/camera", control, methods=["POST"]),
            Route("/checkpoint", checkpoint, methods=["POST"]),
            Route("/blender", blender_page),
            Route("/blender/", blender_page),
            Route("/blender/frame.jpg", blender_frame),
            Route("/blender/camera", blender_control, methods=["POST"]),
            Route("/blender/renderers", blender_renderers),
            Route("/blender/scene", blender_scene, methods=["POST"]),
            Mount(
                "/",
                app=StaticFiles(packages=[("gaussian_splats.visualize", "web")], html=True),
            ),
        ]
    )
    app.state.viewer = GaussianViewer.from_environment()
    try:
        app.state.blender_viewer = BlenderViewer.create()
        app.state.blender_error = None
    except RuntimeError as error:
        app.state.blender_viewer = None
        app.state.blender_error = str(error)
    app.state.render_lock = threading.Lock()
    app.state.pending_controls = []
    app.state.blender_controls = []
    app.state.render_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="viewer")
    return app


def serve(
    checkpoint: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 7007,
) -> None:
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise typer.BadParameter(f"Checkpoint does not exist: {checkpoint}")
        os.environ[CHECKPOINT_ENV] = str(checkpoint)
    else:
        os.environ.pop(CHECKPOINT_ENV, None)

    typer.echo(f"Viewer: http://{host}:{port}")
    uvicorn.run(
        "gaussian_splats.visualize.server:create_app",
        factory=True,
        host=host,
        port=port,
        log_level="warning",
        reload=False,
    )


def main() -> None:
    typer.run(serve)


if __name__ == "__main__":
    main()
