import {
  AppBase, AppOptions, Asset, CameraComponentSystem, Color,
  ContainerHandler, DEVICETYPE_WEBGPU, Entity, FILLMODE_KEEP_ASPECT,
  GAMMA_SRGB, GSPLAT_RENDERER_RASTER_GPU_SORT, GSplatComponentSystem, GSplatHandler,
  TONEMAP_NONE,
  RESOLUTION_FIXED, RenderComponentSystem, TextureHandler, createGraphicsDevice,
} from "playcanvas";
import { addResolution, loadUPlot, resolution } from "/common/components.js";
import { createCameraControls, createHud, request, upload } from "/shared/viewer.js";

const endpoint = document.body.dataset.endpoint;
const form = document.querySelector("form");
const status = document.querySelector("#status");
const lifetime = new AbortController();
const { signal } = lifetime;
addResolution(form);
await loadUPlot();
document.body.classList.add("viewer-container");
const { hud, stats, charts } = createHud(document.body);

const canvas = document.body.appendChild(document.createElement("canvas"));
canvas.className = "viewer-canvas";
canvas.tabIndex = 0;

let app, camera, splat, asset, lastFrame = 0, fps = 0, paused = false, totalSplats = 0;
let stopCamera = () => {};
const values = () => Object.fromEntries(new FormData(form));

async function loadSplat() {
  if (splat) splat.destroy();
  if (asset) {
    app.assets.remove(asset);
    asset.unload();
  }
  const query = new URLSearchParams(values());
  asset = new Asset("checkpoint", "gsplat", { url: `${endpoint}/splats.ply?${query}` });
  app.assets.add(asset);
  await new Promise((resolve, reject) => {
    asset.once("load", resolve);
    asset.once("error", reject);
    app.assets.load(asset);
  });
  splat = new Entity("Gaussian splats");
  splat.addComponent("gsplat", { asset });
  splat.setEulerAngles(-90, 0, 0);
  app.root.addChild(splat);
  const count = Math.min(Number(query.get("num_splats")), totalSplats || Infinity);
  document.querySelector("#splat-total").textContent = `/ ${count.toLocaleString()}`;
  status.textContent = "";
}

function applyCameraSettings() {
  if (!camera) return;
  const settings = values();
  camera.camera.nearClip = Number(settings.near);
  camera.camera.farClip = Number(settings.far);
  const hex = settings.background.slice(1).match(/../g).map(value => parseInt(value, 16) / 255);
  camera.camera.clearColor = new Color(...hex);
}

function applyResolution() {
  const [width, height] = resolution(form);
  app?.setCanvasResolution(RESOLUTION_FIXED, width, height);
  canvas.style.width = canvas.style.height = "100%";
}

function moveCamera({ move, look, dt }) {
  camera.translateLocal(move[0] * dt * 2, move[1] * dt * 2, -move[2] * dt * 2);
  camera.rotate(0, -look[0] * 0.08, 0);
  camera.rotateLocal(look[1] * 0.08, 0, 0);
}

async function main() {
  const checkpoint = await (await request(`${endpoint}/checkpoint`)).json();
  totalSplats = checkpoint.splats;
  form.elements.num_splats.value = totalSplats;
  form.elements.sh_degree.value = Math.min(3, checkpoint.sh_degree ?? 3);
  if (checkpoint.name) status.textContent = checkpoint.name;
  const view = Number(new URLSearchParams(location.search).get("view") || 0) % 4;
  const pose = await (await request(`${endpoint}/camera-info?view=${view}`)).json();
  const device = await createGraphicsDevice(canvas, {
    deviceTypes: [DEVICETYPE_WEBGPU], antialias: false,
  });
  const options = new AppOptions();
  options.graphicsDevice = device;
  options.componentSystems = [RenderComponentSystem, CameraComponentSystem, GSplatComponentSystem];
  options.resourceHandlers = [TextureHandler, ContainerHandler, GSplatHandler];
  app = new AppBase(canvas);
  app.init(options);
  app.setCanvasFillMode(FILLMODE_KEEP_ASPECT);
  applyResolution();
  app.scene.gsplat.renderer = GSPLAT_RENDERER_RASTER_GPU_SORT;
  app.scene.gsplat.useTonemap = false;
  app.scene.gsplat.useFog = false;
  camera = new Entity("Camera");
  camera.addComponent("camera", {
    clearColor: new Color(0, 0, 0),
    fov: pose.fov, nearClip: 0.001, farClip: 100000,
    gammaCorrection: GAMMA_SRGB,
    toneMapping: TONEMAP_NONE,
  });
  camera.setPosition(...pose.position);
  camera.setRotation(...pose.rotation);
  app.root.addChild(camera);
  app.start();
  applyCameraSettings();
  if (checkpoint.name) await loadSplat();
  stopCamera = createCameraControls(
    canvas, hud, moveCamera, signal, error => { status.textContent = error.message; }
  );
  form.addEventListener("change", event => {
    if (event.target.id === "checkpoint") return;
    applyCameraSettings();
    applyResolution();
    if (["num_splats", "sh_degree", "scale", "opacity"].includes(event.target.name)) {
      loadSplat().catch(error => { status.textContent = error.message; });
    }
  }, { signal });
  document.querySelector("#checkpoint").onchange = async event => {
    status.textContent = "Loading checkpoint…";
    const name = await upload(event.target, `${endpoint}/checkpoint`, "X-Checkpoint-Name");
    if (name) {
      status.textContent = name;
      const nextCheckpoint = await (await request(`${endpoint}/checkpoint`)).json();
      totalSplats = nextCheckpoint.splats;
      form.elements.num_splats.value = totalSplats;
      form.elements.sh_degree.value = Math.min(3, nextCheckpoint.sh_degree ?? 3);
      const next = await (await request(`${endpoint}/camera-info?view=${view}`)).json();
      camera.setPosition(...next.position);
      camera.setRotation(...next.rotation);
      await loadSplat();
    }
  };
  document.querySelector("#pause").onclick = event => {
    paused = !paused;
    event.currentTarget.textContent = paused ? "Resume scene" : "Pause scene";
    if (!paused) app.start();
    else app.stop();
  };
  app.on("update", () => {
    const now = performance.now();
    fps = lastFrame ? (fps ? .9 * fps + .1 * 1000 / (now - lastFrame) : 1000 / (now - lastFrame)) : 0;
    lastFrame = now;
    stats.textContent = `FPS ${fps.toFixed(1)}`;
    charts.add(fps, {});
  });
  window.addEventListener("pagehide", () => {
    lifetime.abort();
    stopCamera();
    charts.dispose();
    app.destroy();
  }, { signal });
}

main().catch(error => { status.textContent = error.message; });
