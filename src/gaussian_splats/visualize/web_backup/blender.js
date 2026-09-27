import { bindControls } from "./controls.js";
import { createCharts } from "./charts.js";

const find = selector => document.querySelector(selector);
const viewer = find("#viewer");
const context = viewer.getContext("2d", { alpha: false });
const scene = find("#scene");
const sceneName = find("#scene-name");
const renderer = find("#renderer");
const status = find("#status");
const frame = new Image();
const stats = find("#stats");
const chartWidth = () => Math.min(380, Math.max(160, find("#hud").clientWidth - 24));
const charts = createCharts({ fps: find("#fps-chart"), timing: null, total: null }, chartWidth());
const state = { url: undefined, pending: false, lastFrame: 0, fps: 0 };

const resize = () => {
  viewer.width = Math.max(1, innerWidth);
  viewer.height = Math.max(1, innerHeight);
  context.imageSmoothingEnabled = false;
  charts.resize(chartWidth());
};

const requestFrame = async () => {
  if (state.pending) return;
  state.pending = true;
  try {
    const query = new URLSearchParams({
      width: viewer.width,
      height: viewer.height,
      renderer: renderer.value,
    });
    const response = await fetch(`/blender/frame.jpg?${query}`);
    if (response.status === 204) return;
    if (!response.ok) throw new Error(`Frame request failed: ${response.status}`);
    if (state.url) URL.revokeObjectURL(state.url);
    state.url = URL.createObjectURL(await response.blob());
    frame.src = state.url;
  } catch (error) {
    status.textContent = error.message;
  } finally {
    state.pending = false;
  }
};

frame.onload = () => {
  const now = performance.now();
  const measured = state.lastFrame ? 1000 / (now - state.lastFrame) : 0;
  state.fps = state.lastFrame ? 0.9 * state.fps + 0.1 * measured : measured;
  state.lastFrame = now;
  context.drawImage(frame, 0, 0, viewer.width, viewer.height);
  stats.textContent = `FPS: ${state.fps.toFixed(1)}`;
  charts.add(state.fps);
  requestAnimationFrame(requestFrame);
};

const loadScene = async () => {
  const file = scene.files[0];
  if (!file) return;
  scene.disabled = true;
  sceneName.textContent = file.name;
  try {
    const response = await fetch("/blender/scene", {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream", "X-Scene-Name": file.name },
      body: file,
    });
    if (!response.ok) {
      const result = await response.json();
      throw new Error(result.error || `Upload failed: ${response.status}`);
    }
    status.textContent = file.name;
    requestFrame();
  } catch (error) {
    sceneName.textContent = "No file selected";
    status.textContent = error.message;
  } finally {
    scene.disabled = false;
    scene.value = "";
  }
};

const loadRenderers = async () => {
  const response = await fetch("/blender/renderers");
  if (!response.ok) return;
  for (const name of await response.json()) renderer.add(new Option(name, name));
};

window.onresize = resize;
scene.onchange = loadScene;
renderer.onchange = requestFrame;
bindControls(viewer, find("#hud"), "/blender/camera");
resize();
loadRenderers().then(requestFrame);
