import { createCharts } from "./charts.js";
import { bindControls } from "./controls.js";

const find = selector => document.querySelector(selector);
const viewer = find("#viewer");
const context = viewer.getContext("2d", { alpha: false });
const settings = find("#settings form");
const checkpoint = find("#checkpoint");
const checkpointName = find("#checkpoint-name");
const stats = find("#stats");
const splatTotal = find("#splat-total");
const frame = new Image();
const state = {
  fps: 0,
  lastFrame: 0,
  rtt: undefined,
  timings: undefined,
  url: undefined,
  pending: false,
};
const chartWidth = () => Math.min(420, Math.max(160, innerWidth - 64));
const charts = createCharts({
  fps: find("#fps-chart"),
  timing: find("#timing-chart"),
  total: find("#timing-total"),
}, chartWidth());

const retry = () => setTimeout(requestFrame, 1000);

const resize = () => {
  const tile = Number(settings.elements.tile_size.value);
  viewer.height = tile * Math.max(1, Math.round(480 / tile));
  viewer.width = Math.min(
    1280,
    tile * Math.max(1, Math.round(viewer.height * innerWidth / innerHeight / tile)),
  );
  context.imageSmoothingEnabled = false;
  charts.resize(chartWidth());
};

async function requestFrame() {
  if (state.pending) return;
  state.pending = true;
  try {
    const query = new URLSearchParams(new FormData(settings));
    query.set("width", viewer.width);
    query.set("height", viewer.height);
    const response = await fetch(`/frame.jpg?${query}`);
    if (response.status === 204) {
      stats.textContent = "No checkpoint loaded";
      return;
    }
    if (!response.ok) throw new Error(`Frame request failed: ${response.status}`);
    state.timings = JSON.parse(response.headers.get("X-Timings") || "null");
    const selectedName = response.headers.get("X-Checkpoint-Name");
    if (selectedName) {
      checkpointName.textContent = selectedName;
      checkpointName.title = selectedName;
    }
    const total = Number(response.headers.get("X-Total-Splats"));
    splatTotal.textContent = `/ ${total.toLocaleString()}`;
    settings.elements.num_splats.max = total;
    if (state.url) URL.revokeObjectURL(state.url);
    state.url = URL.createObjectURL(await response.blob());
    frame.src = state.url;
  } catch (error) {
    console.error(error);
    retry();
  } finally {
    state.pending = false;
  }
}

frame.onload = () => {
  const now = performance.now();
  const measured = state.lastFrame ? 1000 / (now - state.lastFrame) : 0;
  state.fps = state.lastFrame ? 0.9 * state.fps + 0.1 * measured : measured;
  state.lastFrame = now;
  context.drawImage(frame, 0, 0, viewer.width, viewer.height);
  stats.textContent = `RTT: ${state.rtt?.toFixed(0) ?? "--"} ms`;
  charts.add(state.fps, state.timings?.default);
  requestAnimationFrame(requestFrame);
};
frame.onerror = retry;

const ping = async () => {
  const start = performance.now();
  try {
    if ((await fetch("/ping")).ok) state.rtt = performance.now() - start;
  } catch (error) {
    console.error(error);
  }
};

const showRelevantSettings = () => {
  const hidden = settings.elements.rendering_mode.value !== "ellipsoid";
  find("#confidence-setting").hidden = hidden;
  settings.elements.confidence.disabled = hidden;
};

const loadCheckpoint = async () => {
  const file = checkpoint.files[0];
  if (!file) return;
  checkpoint.disabled = true;
  checkpointName.textContent = file.name;
  checkpointName.title = file.name;
  try {
    const response = await fetch("/checkpoint", {
      method: "POST",
      headers: {
        "Content-Type": "application/octet-stream",
        "X-Checkpoint-Name": file.name,
      },
      body: file,
    });
    if (!response.ok) {
      const result = await response.json();
      throw new Error(result.error || `Upload failed: ${response.status}`);
    }
    settings.elements.num_splats.value = "10000";
    requestFrame();
  } catch (error) {
    console.error(error);
  } finally {
    checkpoint.disabled = false;
    checkpoint.value = "";
  }
};

window.onresize = resize;
settings.elements.tile_size.onchange = resize;
settings.elements.rendering_mode.onchange = showRelevantSettings;
checkpoint.onchange = loadCheckpoint;
setInterval(ping, 1000);
bindControls(viewer, find("#hud"));
showRelevantSettings();
resize();
ping();
requestFrame();
