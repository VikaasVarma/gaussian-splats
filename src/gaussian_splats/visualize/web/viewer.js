import { createCharts } from "./charts.js";

export async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`${url}: ${response.status}\n${await response.text()}`);
  return response;
}

export function post(url, data, signal) {
  return request(url, {
    method: "POST", signal,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

export async function upload(input, url, header) {
  const file = input.files[0];
  if (!file) return false;
  input.disabled = true;
  try {
    await request(url, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream", [header]: file.name },
      body: file,
    });
    return file.name;
  } finally {
    input.disabled = false;
  }
}

function cameraControls(canvas, hud, endpoint, signal, fail) {
  const listen = (target, event, handler) => target.addEventListener(event, handler, { signal });
  const keys = new Set();
  let look = [0, 0], movement = [0, 0, 0], elapsed = 0;
  let sampled = performance.now(), cameraPending = false;

  function sample() {
    const now = performance.now(), dt = (now - sampled) / 1000;
    sampled = now;
    const direction = [
      Number(keys.has("KeyD")) - Number(keys.has("KeyA")),
      Number(keys.has("Space")) - Number(keys.has("ShiftLeft") || keys.has("ShiftRight")),
      Number(keys.has("KeyW")) - Number(keys.has("KeyS")),
    ];
    const length = Math.max(1, Math.hypot(...direction));
    movement = movement.map((value, axis) => value + direction[axis] / length * dt);
    elapsed += dt;
  }

  function release() {
    keys.clear();
    look = [0, 0];
    movement = [0, 0, 0];
    elapsed = 0;
    sampled = performance.now();
  }

  listen(canvas, "click", () => { canvas.focus(); canvas.requestPointerLock(); });
  listen(window, "keydown", event => {
    if (event.target.closest("input, select, textarea, button, [contenteditable]")) return;
    if (event.code === "KeyH" && !event.repeat) hud.open = !hud.open;
    if (document.activeElement !== canvas && document.pointerLockElement !== canvas) return;
    if (["KeyW", "KeyA", "KeyS", "KeyD", "Space", "ShiftLeft", "ShiftRight"].includes(event.code)) {
      event.preventDefault();
      sample();
      keys.add(event.code);
    }
  });
  listen(window, "keyup", event => { sample(); keys.delete(event.code); });
  listen(window, "blur", release);
  listen(canvas, "blur", release);
  listen(document, "pointerlockchange", () => {
    if (document.pointerLockElement !== canvas) release();
  });
  listen(document, "visibilitychange", release);
  listen(window, "mousemove", event => {
    if (document.pointerLockElement === canvas) {
      look[0] += event.movementX;
      look[1] -= event.movementY;
    }
  });

  const cameraTimer = setInterval(() => {
    sample();
    if (cameraPending || (!movement.some(Boolean) && !look.some(Boolean))) return;
    const data = { move: movement.map(value => value / elapsed), look, dt: elapsed };
    movement = [0, 0, 0];
    look = [0, 0];
    elapsed = 0;
    cameraPending = true;
    post(`${endpoint}/camera`, data, signal)
      .finally(() => { cameraPending = false; }).catch(fail);
  }, 50);

  return () => {
    clearInterval(cameraTimer);
    if (document.pointerLockElement === canvas) document.exitPointerLock();
  };
}

export function createViewer({
  endpoint, settings = () => ({}), container = document.body,
  canvas = container.appendChild(document.createElement("canvas")),
  size = () => [Math.min(1280, canvas.clientWidth), canvas.clientHeight],
  emptyMessage = "Load a scene", onFrame = () => {},
}) {
  container.classList.add("viewer-container");
  canvas.classList.add("viewer-canvas");
  canvas.tabIndex = 0;
  const context = canvas.getContext("2d", { alpha: false });
  const hud = document.createElement("details");
  hud.className = "panel viewer-hud";
  hud.open = true;
  hud.innerHTML = `<summary>Statistics</summary><div class="stats"></div>
    <div class="fps-chart"></div><div class="timing-chart"></div>
    <div class="frame-status"></div>
    <footer>Click to look · WASD to move · Space/Shift up/down · H toggles HUD</footer>`;
  container.append(hud);
  const stats = hud.querySelector(".stats");
  const status = hud.querySelector(".frame-status");
  const charts = createCharts(hud);
  const lifetime = new AbortController();
  const { signal } = lifetime;
  let lastFrame = 0, fps = 0, animation, emptyTimer, paused = false;
  const stopCamera = cameraControls(canvas, hud, endpoint, signal, fail);
  if (typeof settings !== "function") {
    const form = settings;
    let values = Object.fromEntries(new FormData(form));
    form.addEventListener("change", () => {
      if (form.checkValidity()) values = Object.fromEntries(new FormData(form));
    }, { signal });
    settings = () => values;
  }

  function fail(error) {
    if (signal.aborted) return;
    status.textContent = error.message;
    dispose();
    throw error;
  }

  async function render() {
    if (paused) return;
    const [width, height] = size();
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const params = new URLSearchParams({ ...settings(), width, height });
    const started = performance.now();
    const response = await request(`${endpoint}/frame.jpg?${params}`, { signal });
    if (paused) return;
    if (response.status === 204) {
      if (paused) return;
      status.textContent = emptyMessage;
      lastFrame = 0;
      emptyTimer = setTimeout(() => render().catch(fail), 500);
      return;
    }
    const bitmap = await createImageBitmap(await response.blob());
    if (signal.aborted) { bitmap.close(); return; }
    context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    bitmap.close();
    const now = performance.now();
    fps = lastFrame ? (fps ? .9 * fps + .1 * 1000 / (now - lastFrame) : 1000 / (now - lastFrame)) : 0;
    lastFrame = now;
    const milliseconds = now - started;
    const timings = JSON.parse(response.headers.get("X-Timings"));
    charts.add(fps, timings);
    stats.textContent = `FPS ${fps.toFixed(1)} · Frame ${milliseconds.toFixed(0)} ms`;
    status.textContent = "";
    onFrame({ response, fps, milliseconds, timings });
    if (!paused) animation = requestAnimationFrame(() => render().catch(fail));
  }

  function pause() {
    paused = true;
  }

  function resume() {
    if (!paused) return;
    paused = false;
    render().catch(fail);
  }

  function dispose() {
    lifetime.abort();
    stopCamera();
    clearTimeout(emptyTimer);
    cancelAnimationFrame(animation);
    charts.dispose();
  }

  window.addEventListener("pagehide", dispose, { signal });
  render().catch(fail);
  return { canvas, hud, pause, resume, dispose };
}
