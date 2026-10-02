# Viewer scratchpads

Run `uv run visualize` from the project root. `/` is a small page registry;
`/gaussian` shows checkpoints;
`/scene` and `/blender` show Blender/Cycles/Workbench; `/bake` bakes Workbench
splats and renders both views with one camera. Existing frame, camera, checkpoint,
scene, renderer-list, and bake URLs remain available. Page URLs redirect to a
trailing slash so their relative scripts resolve within the page folder.

Each page owns `backend.py`, `index.html`, and a short `page.js` when it needs
browser behavior. `server.py` owns HTTP and the render worker; `camera.py` owns
camera math; `web/` supplies the viewer, charts, and styles; `common/` supplies
small reusable browser components such as uPlot loading and resolution controls.
There is no UI schema, page inheritance, discovery, or build step.

## Add a page

Create `experiment/__init__.py`, `experiment/backend.py`, `experiment/index.html`,
and (if needed) `experiment/page.js`. The backend needs `controller` (a `CameraController` or
`None`) and `render(settings, width, height)`. Return a `Frame` with an H×W×3 float
RGB tensor, optional flat named timings in milliseconds, and optional string HTTP
metadata headers. Return `None` while waiting for a scene. Add `close()` only if
the backend owns resources.

```python
from pathlib import Path

import torch

from gaussian_splats.visualize import Frame
from gaussian_splats.visualize.server import Page


class Experiment:
    controller = None

    def render(self, settings, width, height):
        image = torch.full((height, width, 3), float(settings["brightness"]))
        return Frame(image)


def page():
    return Page(Experiment, Path(__file__).parent)
```

Import that `page` factory in `create_app()` and add one registration:
`(("/experiment",), experiment_page())`. This installs the HTML page,
`/experiment/frame.jpg`, and `/experiment/camera`.

```html
<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experiment</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/uplot@1.6.32/dist/uPlot.min.css">
<link rel="stylesheet" href="/shared/styles.css">
<details id="settings" class="panel" open>
  <summary>Experiment</summary>
  <form>
    <label>Brightness <input name="brightness" type="number" value="0.5" step="0.1"></label>
  </form>
</details>
<script src="https://cdn.jsdelivr.net/npm/uplot@1.6.32/dist/uPlot.iife.min.js"></script>
<script type="module" src="page.js"></script>
</html>
```

`page.js` can import `/shared/viewer.js` and own any controls or actions unique
to the experiment.

## Page-specific operations

Add `(path, async_handler, methods)` tuples to `page.extra_routes`. Use
`await page.run(function, *args)` for **all** renderer state reads and mutations,
including custom actions. Startup calls `page.start(worker)` to construct each
backend on the shared worker, which also moves
cameras, renders/encodes frames, and closes resources. There are no additional
locks or control queues. Backends are shared between tabs, as before.

`upload(request, page, loader, filename_header, allowed_suffixes)` streams a file
to temporary storage, calls `loader(path, original_filename)` on the worker, and
deletes the upload afterward, including on failure. If an experiment needs the
file after loading, its loader must copy it into storage owned by that backend.

Browser helpers exported by `viewer.js`:

- `request(url, options)` returns a checked Fetch response.
- `post(url, data)` sends JSON and returns a checked response.
- `upload(input, url, filenameHeader)` disables the input while sending its file,
  leaves the selected filename visible, and returns the filename (or `false` when
  no file was selected).
- `createViewer(options)` creates the canvas/HUD and starts rendering and input.
  `settings` accepts a form (valid changes are committed on `change`) or a
  function returning query parameters on each frame. Optional `canvas` and
  a sized `container` support custom layouts; `size()` returns render width
  and height. `onFrame({response, fps, milliseconds, timings})` handles custom
  metadata. The result exposes `canvas`, `hud`, and `dispose()`.

The render loop owns scheduling; changing settings needs no manual render call.
Mark numeric controls `required` to keep empty edits uncommitted. Programmatic
form updates should dispatch `change`. The default size follows the canvas layout.
Camera requests run independently with at most one outstanding request and
accumulated movement using elapsed time. `dispose()` stops requests, listeners,
timers, and charts; it leaves the canvas/HUD elements available to the caller.
Keyboard controls activate on the focused canvas or pointer lock, ignore form
editing, and clear on focus loss.

Unexpected failures produce development tracebacks and rejected browser promises;
frame/camera failures stop the viewer and appear in its HUD. There is no retry
loop or silent chart fallback. The chart library still loads from the existing
pinned CDN. Timings are separate series, not stacked: summing potentially nested
measurements can misrepresent total render time. The HUD shows end-to-end frame
latency separately.

## Verification

```sh
uv run --with httpx pytest src/gaussian_splats/visualize/tests -q
uv run ruff check src/gaussian_splats/visualize
```

The suite checks routes/assets, checkpoint rendering and settings, timing values,
camera movement, a fourth scratchpad with a custom action, serialized operations,
temporary-file cleanup, and visible failures. With Blender installed it also
checks all four renderers and scene loading/baking. Browser behavior
requires a browser smoke test; Python endpoint tests do not cover pointer lock,
DOM layout, charts, or JavaScript lifecycle.
