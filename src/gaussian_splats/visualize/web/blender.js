import { createCharts } from "./charts.js";
import { bindControls } from "./controls.js";

const $ = selector => document.querySelector(selector);
const canvas = $("#viewer");
const context = canvas.getContext("2d", { alpha: false });
const scene = $("#scene");
const sceneName = $("#scene-name");
const renderer = $("#renderer");
const status = $("#status");
const stats = $("#stats");
const image = new Image();
const charts = createCharts({ fps: $("#fps-chart") }, 360);
const state = { request: null, lastFrame: 0, fps: 0, paused: false };

function resize() {
  canvas.width = Math.min(1280, Math.max(1, innerWidth));
  canvas.height = Math.max(1, innerHeight);
  charts.resize(Math.min(420, Math.max(180, innerWidth - 48)));
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
  if (state.request || state.paused || !renderer.value) return;
  state.request = fetch(`/blender/frame.jpg?width=${canvas.width}&height=${canvas.height}&renderer=${renderer.value}`);
  try {
    const response = await state.request;
    if (response.status === 204) {
      status.textContent = "Choose a glTF or GLB scene";
      await new Promise(resolve => setTimeout(resolve, 500));
      return;
    }
    if (!response.ok) throw new Error(`Render failed (${response.status})`);
    const now = performance.now();
    const frameTime = state.lastFrame ? now - state.lastFrame : 0;
    state.fps = frameTime ? (state.fps ? state.fps * 0.9 + 1000 / frameTime * 0.1 : 1000 / frameTime) : 0;
    state.lastFrame = now;
    stats.textContent = `FPS ${state.fps.toFixed(1)}`;
    charts.add(state.fps);
    await drawFrame(await response.blob());
  } catch (error) {
    status.textContent = error.message;
    await new Promise(resolve => setTimeout(resolve, 1000));
  } finally {
    state.request = null;
    requestAnimationFrame(render);
  }
}

async function loadScene() {
  const file = scene.files[0];
  if (!file) return;
  state.paused = true;
  scene.disabled = true;
  sceneName.textContent = file.name;
  status.textContent = "Loading scene…";
  try {
    const response = await fetch("/blender/scene", {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream", "X-Scene-Name": file.name },
      body: file,
    });
    if (!response.ok) throw new Error(`Scene load failed (${response.status})`);
    status.textContent = "Ready";
  } catch (error) {
    sceneName.textContent = "No file selected";
    sceneName.removeAttribute("title");
    status.textContent = error.message;
  } finally {
    scene.value = "";
    scene.disabled = false;
    state.paused = false;
    render();
  }
}

async function loadRenderers() {
  const response = await fetch("/blender/renderers");
  if (!response.ok) throw new Error("Blender viewer unavailable");
  for (const name of await response.json()) renderer.add(new Option(name, name));
}

scene.addEventListener("change", loadScene);
renderer.addEventListener("change", render);
window.addEventListener("resize", resize);
bindControls(canvas, { endpoint: "/blender/camera", hud: $("#hud") });
resize();
loadRenderers().then(render).catch(error => { status.textContent = error.message; });
