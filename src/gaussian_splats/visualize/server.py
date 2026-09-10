"""Serve rasterizer frames to the browser viewer."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

from PIL import Image

from .rasterizer import rasterize

_VIEWER = Path(__file__).with_name("viewer.html").read_bytes()


def render_frame(width: int = 640, height: int = 480) -> bytes:
    output = BytesIO()
    Image.fromarray(rasterize(width, height), "RGB").save(output, "JPEG", quality=75)
    return output.getvalue()


class ViewerHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = self.path.partition("?")[0]
        if path in ("/", "/index.html"):
            self._send(_VIEWER, "text/html; charset=utf-8")
        elif path == "/frame.jpg":
            self._send(render_frame(), "image/jpeg")
        else:
            self.send_error(404)

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7007)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    print(f"Viewer running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
