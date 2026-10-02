import { createViewer, post, request, upload } from "/shared/viewer.js";
import { addResolution, loadUPlot, resolution } from "/common/components.js";

const form = document.querySelector("form");
const status = document.querySelector("#status");
const endpoint = document.body.dataset.endpoint;
addResolution(form);
await loadUPlot();
const checkpoint = await (await request(`${endpoint}/checkpoint`)).json();
if (checkpoint.name) status.textContent = checkpoint.name;

createViewer({
  endpoint,
  settings: form,
  emptyMessage: "Choose a checkpoint",
  size: () => resolution(form),
  onFrame: ({ response }) => {
    const count = Number(response.headers.get("X-Total-Splats"));
    const input = form.elements.num_splats;
    if (input.max !== String(count)) {
      input.max = count;
      input.value = Math.min(Number(input.value), count);
      form.dispatchEvent(new Event("change"));
    }
    document.querySelector("#splat-total").textContent = `/ ${count.toLocaleString()}`;
  },
});

form.elements.rendering_mode.onchange = () => {
  const ellipsoid = form.elements.rendering_mode.value === "ellipsoid";
  form.elements.confidence.disabled = !ellipsoid;
  document.querySelector("#confidence-setting").hidden = !ellipsoid;
};

document.querySelector("#checkpoint").onchange = async event => {
  status.textContent = "Loading checkpoint…";
  const name = await upload(event.target, `${endpoint}/checkpoint`, "X-Checkpoint-Name");
  if (name) {
    status.textContent = name;
  }
};

document.querySelector("#clear").onclick = async () => {
  await post(`${endpoint}/checkpoint/clear`, {});
  document.querySelector("#checkpoint").value = "";
  document.querySelector("#splat-total").textContent = "/ --";
  status.textContent = "Choose a checkpoint";
};
