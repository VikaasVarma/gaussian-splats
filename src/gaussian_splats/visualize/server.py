"""Shared HTTP transport for small, independent viewer pages."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, asynccontextmanager
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

import typer
import uvicorn
from PIL import Image
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles


class Page:
    def __init__(self, factory, directory: Path):
        self.factory = factory
        self.directory = directory
        self.extra_routes = []

    async def start(self, worker: ThreadPoolExecutor) -> None:
        self.worker = worker
        self.backend = await self.run(self.factory)

    async def run(self, function, *args):
        return await asyncio.wrap_future(self.worker.submit(function, *args))

    async def frame(self, request):
        params = dict(request.query_params)
        width, height = int(params.pop("width", 640)), int(params.pop("height", 480))

        def render():
            frame = self.backend.render(params, width, height)
            if frame is None:
                return Response(status_code=204, headers={"X-No-Scene": "1"})
            output = BytesIO()
            pixels = (255 * frame.image.clamp(0, 1)).byte().cpu().numpy()
            Image.fromarray(pixels).save(output, format="JPEG", quality=75)
            return Response(
                output.getvalue(),
                media_type="image/jpeg",
                headers={
                    "Cache-Control": "no-store",
                    "X-Timings": json.dumps(frame.timings),
                    **frame.metadata,
                },
            )

        return await self.run(render)

    async def camera(self, request):
        data = await request.json()

        def move():
            if self.backend.controller is not None:
                self.backend.controller.update(data["move"], data["look"], float(data["dt"]))

        await self.run(move)
        return Response(status_code=204)

    def script(self, request):
        return FileResponse(self.directory / "page.js")

    def routes(self, prefix):
        async def index(request):
            return FileResponse(self.directory / "index.html")

        routes = [
            Route(prefix + "/", index),
            Route(prefix + "/frame.jpg", self.frame),
            Route(prefix + "/camera", self.camera, methods=["POST"]),
        ]
        if (self.directory / "page.js").is_file():
            routes.append(Route(prefix + "/page.js", self.script))
        routes.extend(
            Route(prefix + path, handler, methods=methods)
            for path, handler, methods in self.extra_routes
        )
        return routes


async def upload(request, page, load, header, suffixes):
    filename = Path(request.headers[header]).name
    suffix = Path(filename).suffix.lower()
    if suffix not in suffixes:
        return JSONResponse({"error": f"Expected {', '.join(suffixes)}"}, status_code=415)
    with NamedTemporaryFile(suffix=suffix) as temporary:
        async for chunk in request.stream():
            temporary.write(chunk)
        temporary.flush()
        await page.run(load, Path(temporary.name), filename)
    return Response(status_code=204)


def create_app(registrations=None):
    if registrations is None:
        from .bake.backend import page as bake_page
        from .compare.backend import page as compare_page
        from .gaussian.backend import page as gaussian_page

        registrations = [
            (("/gaussian",), gaussian_page()),
            (("/bake", "/blender"), bake_page()),
            (("/compare",), compare_page()),
        ]

    @asynccontextmanager
    async def lifespan(app):
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="viewer") as worker:
            cleanup = ExitStack()
            try:
                for _, page in registrations:
                    await page.start(worker)
                    close = getattr(page.backend, "close", None)
                    if close is not None:
                        cleanup.callback(close)
                yield
            finally:
                await asyncio.wrap_future(worker.submit(cleanup.close))

    async def ping(request):
        return Response(status_code=204)

    home = Path(__file__).parent / "home" / "index.html"

    async def index(request):
        return FileResponse(home)

    routes = [Route("/", index), Route("/ping", ping)]
    for prefixes, page in registrations:
        for prefix in prefixes:
            routes.extend(page.routes(prefix))
    routes.append(Mount("/shared", StaticFiles(directory=Path(__file__).parent / "web")))
    routes.append(Mount("/common", StaticFiles(directory=Path(__file__).parent / "common")))
    return Starlette(debug=True, routes=routes, lifespan=lifespan)


def serve(host: str = "127.0.0.1", port: int = 7007):
    typer.echo(f"Viewer: http://{host}:{port}")
    uvicorn.run(
        "gaussian_splats.visualize.server:create_app",
        factory=True,
        host=host,
        port=port,
        log_level="info",
    )


def main():
    typer.run(serve)


if __name__ == "__main__":
    main()
