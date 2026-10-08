import { createViewer, post, request, upload } from "/shared/viewer.js";
import { addResolution, loadUPlot, resolution, renderSettings } from "/common/components.js";

const form = document.querySelector("form");
const status = document.querySelector("#status");
const bake = document.querySelector("#bake");
const rendererSettings = document.querySelector("#renderer-settings");
const endpoint = document.body.dataset.endpoint;
addResolution(form);
const renderers = await (await request(`${endpoint}/renderers`)).json();
for (const renderer of renderers) {
  form.elements.renderer.add(new Option(renderer.label, renderer.value));
}
form.elements.renderer.value = "torch-workbench";

function showRendererSettings() {
  const renderer = renderers.find(item => item.value === form.elements.renderer.value);
  renderSettings(rendererSettings, renderer.settings);
  document.querySelector("#left-label").textContent = renderer.label;
  bake.textContent = `Bake ${renderer.label} → Splats`;
}

form.elements.renderer.addEventListener("change", showRendererSettings);
showRendererSettings();
await loadUPlot();

const viewer = createViewer({
  endpoint,
  settings: form,
  size: () => resolution(form),
  emptyMessage: "Choose a glTF or GLB scene",
  onFrame: ({ response }) => {
    document.querySelector("#count").textContent =
      `${Number(response.headers.get("X-Total-Splats")).toLocaleString()} splats`;
  },
});

document.querySelector("#scene").onchange = async event => {
  bake.disabled = true;
  status.textContent = "Loading scene…";
  const name = await upload(event.target, `${endpoint}/scene`, "X-Scene-Name");
  if (name) {
    bake.disabled = false;
    status.textContent = `${name} · Ready to bake`;
  }
};

bake.onclick = async () => {
  if (!form.reportValidity()) return;
  bake.disabled = true;
  status.textContent = "Baking…";
  try {
    const result = await (await post(`${endpoint}/bake`, {
      n_splats: Number(form.elements.num_splats.value),
      ...Object.fromEntries(new FormData(form)),
    })).json();
    status.textContent = `Ready · ${result.splats.toLocaleString()} splats`;
  } catch (error) {
    status.textContent = error.message;
  } finally {
    bake.disabled = false;
  }
};

document.querySelector("#pause").onclick = event => {
  const button = event.currentTarget;
  if (button.textContent === "Pause scene") {
    viewer.pause();
    button.textContent = "Resume scene";
    status.textContent = "Paused";
  } else {
    viewer.resume();
    button.textContent = "Pause scene";
    status.textContent = "";
  }
};
