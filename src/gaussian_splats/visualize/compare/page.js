import { createViewer, post, upload } from "/shared/viewer.js";
import { addResolution, loadUPlot, resolution } from "/common/components.js";

const form = document.querySelector("form");
const status = document.querySelector("#status");
const bake = document.querySelector("#bake");
const endpoint = document.body.dataset.endpoint;
addResolution(form);
await loadUPlot();

createViewer({
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
  bake.disabled = true;
  status.textContent = "Baking…";
  try {
    const result = await (await post(`${endpoint}/bake`, {
      n_splats: Number(form.elements.num_splats.value),
    })).json();
    status.textContent = `Ready · ${result.splats.toLocaleString()} splats`;
  } finally {
    bake.disabled = false;
  }
};

document.querySelector("#clear").onclick = async () => {
  await post(`${endpoint}/scene/clear`, {});
  document.querySelector("#scene").value = "";
  bake.disabled = true;
  document.querySelector("#count").textContent = "";
  status.textContent = "Choose a glTF or GLB scene";
};
