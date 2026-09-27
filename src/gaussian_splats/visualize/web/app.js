import { createCharts } from "./charts.js";
import { bindControls } from "./controls.js";

const $ = selector => document.querySelector(selector);
const canvas = $("#viewer");
const context = canvas.getContext("2d", { alpha: false });
const form = $("#settings form");
const checkpoint = $("#checkpoint");
const checkpointName = $("#checkpoint-name");
const status = $("#status");
const stats = $("#stats");
const total = $("#splat-total");
const image = new Image();
const charts = createCharts({ fps: $("#fps-chart"), timing: $("#timing-chart"), total: $("#timing-total") }, 360);
const state = { request: null, lastFrame: 0, fps: 0, rtt: 0, paused: false };

const wait = delay => new Promise(resolve => setTimeout(resolve, delay));
const chartWidth = () => Math.min(420, Math.max(180, innerWidth - 48));

function resize() {
  const tile = Number(form.elements.tile_size.value) || 8;
  const height = tile * Math.max(1, Math.round(480 / tile));
  canvas.height = height;
  canvas.width = Math.min(1280, tile * Math.max(1, Math.round(height * innerWidth / innerHeight / tile)));
  charts.resize(chartWidth());
}

async function drawFrame(blob) {
  const url = URL.createObjectURL(blob);
  try {
    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = reject;
      image.src = url;
    });
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function render() {
  if (state.request || state.paused) return;
  const started = performance.now();
  state.request = fetch(`/frame.jpg?${new URLSearchParams(new FormData(form))}&width=${canvas.width}&height=${canvas.height}`);
  try {
    const response = await state.request;
    if (response.status === 204) {
      status.textContent = "Choose a checkpoint";
      await wait(500);
      return;
    }
    if (!response.ok) throw new Error(`Render failed (${response.status})`);
    const now = performance.now();
    const frameTime = state.lastFrame ? now - state.lastFrame : 0;
    state.fps = frameTime ? (state.fps ? state.fps * 0.9 + 1000 / frameTime * 0.1 : 1000 / frameTime) : 0;
    state.lastFrame = now;
    state.rtt = performance.now() - started;
    const count = Number(response.headers.get("X-Total-Splats") || 0);
    total.textContent = count ? `/ ${count.toLocaleString()}` : "/ --";
    form.elements.num_splats.max = count || 1;
    charts.add(state.fps, JSON.parse(response.headers.get("X-Timings") || "null"));
    stats.textContent = `FPS ${state.fps.toFixed(1)} · ${state.rtt ? `${state.rtt.toFixed(0)} ms` : "rendering"}`;
    await drawFrame(await response.blob());
    status.textContent = checkpointName.textContent === "No file selected" ? "No checkpoint loaded" : "Ready";
  } catch (error) {
    status.textContent = error.message;
    await wait(1000);
  } finally {
    state.request = null;
    requestAnimationFrame(render);
  }
}

async function loadCheckpoint() {
  const file = checkpoint.files[0];
  if (!file) return;
  state.paused = true;
  checkpoint.disabled = true;
  checkpointName.textContent = file.name;
  checkpointName.title = file.name;
  status.textContent = "Loading checkpoint…";
  try {
    const response = await fetch("/checkpoint", {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream", "X-Checkpoint-Name": file.name },
      body: file,
    });
    if (!response.ok) throw new Error(`Checkpoint load failed (${response.status})`);
    form.elements.num_splats.value = "10000";
    status.textContent = "Ready";
  } catch (error) {
    checkpointName.textContent = "No file selected";
    checkpointName.removeAttribute("title");
    status.textContent = error.message;
  } finally {
    checkpoint.value = "";
    checkpoint.disabled = false;
    state.paused = false;
    render();
  }
}

form.addEventListener("change", event => {
  if (event.target === checkpoint) return;
  if (event.target === form.elements.tile_size) resize();
  if (event.target === form.elements.rendering_mode) {
    const enabled = form.elements.rendering_mode.value === "ellipsoid";
    form.elements.confidence.disabled = !enabled;
    $("#confidence-setting").hidden = !enabled;
  }
  render();
});
checkpoint.addEventListener("change", loadCheckpoint);
window.addEventListener("resize", resize);
bindControls(canvas, { endpoint: "/camera", hud: $("#hud") });
resize();
render();
