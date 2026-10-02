import { createViewer, post, request, upload } from "/shared/viewer.js";
import { addResolution, loadUPlot, resolution } from "/common/components.js";

const form = document.querySelector("form");
const status = document.querySelector("#status");
const endpoint = document.body.dataset.endpoint;
const rendererSettings = document.querySelector("#renderer-settings");
addResolution(form);
const renderers = await (await request(`${endpoint}/renderers`)).json();
for (const renderer of renderers) {
  form.elements.renderer.add(new Option(renderer.label, renderer.value));
}

function showRendererSettings() {
  const renderer = renderers.find(item => item.value === form.elements.renderer.value);
  rendererSettings.replaceChildren();
  for (const setting of renderer?.settings ?? []) {
    const label = document.createElement("label");
    label.textContent = setting.label;
    const input = document.createElement("input");
    input.name = setting.name;
    input.type = "number";
    input.value = setting.value;
    input.min = setting.min;
    input.step = setting.step;
    label.append(input);
    rendererSettings.append(label);
  }
}

form.elements.renderer.addEventListener("change", showRendererSettings);
showRendererSettings();
await loadUPlot();
createViewer({
  endpoint,
  settings: form,
  size: () => resolution(form),
  emptyMessage: "Choose a glTF or GLB scene",
});

document.querySelector("#scene").onchange = async event => {
  status.textContent = "Loading scene…";
  const name = await upload(event.target, `${endpoint}/scene`, "X-Scene-Name");
  if (name) status.textContent = name;
};

document.querySelector("#clear").onclick = async () => {
  await post(`${endpoint}/scene/clear`, {});
  document.querySelector("#scene").value = "";
  status.textContent = "Choose a glTF or GLB scene";
};
