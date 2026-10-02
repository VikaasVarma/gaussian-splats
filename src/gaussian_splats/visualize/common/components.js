let uPlot;

export function loadUPlot() {
  if (uPlot) return Promise.resolve(uPlot);
  uPlot = new Promise((resolve, reject) => {
    const stylesheet = document.createElement("link");
    stylesheet.rel = "stylesheet";
    stylesheet.href = "https://cdn.jsdelivr.net/npm/uplot@1.6.32/dist/uPlot.min.css";
    document.head.append(stylesheet);

    const script = document.createElement("script");
    script.src = "https://cdn.jsdelivr.net/npm/uplot@1.6.32/dist/uPlot.iife.min.js";
    script.onload = () => resolve(window.uPlot);
    script.onerror = reject;
    document.head.append(script);
  });
  return uPlot;
}

export function addResolution(form, initial = "640x480") {
  const label = document.createElement("label");
  label.innerHTML = `Resolution <select name="resolution">
    <option value="640x480">640 × 480</option>
    <option value="1280x720">1280 × 720</option>
    <option value="1920x1080">1920 × 1080</option>
  </select>`;
  form.prepend(label);
  form.elements.resolution.value = initial;
}

export function resolution(form) {
  return form.elements.resolution.value.split("x").map(Number);
}
