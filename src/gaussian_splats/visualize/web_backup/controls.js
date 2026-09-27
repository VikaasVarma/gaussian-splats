export const bindControls = (viewer, hud, endpoint = "/camera") => {
  const keys = new Set();
  const axis = (positive, negative) => Number(keys.has(positive)) - Number(keys.has(negative));
  let look = [0, 0];

  viewer.onclick = () => viewer.requestPointerLock();
  document.onkeydown = event => {
    keys.add(event.code);
    if (event.code === "KeyH" && !event.repeat && hud) hud.open = !hud.open;
  };
  document.onkeyup = event => keys.delete(event.code);
  document.onmousemove = event => {
    if (document.pointerLockElement !== viewer) return;
    look[0] += event.movementX;
    look[1] += event.movementY;
  };

  setInterval(() => {
    const vertical = Number(keys.has("Space")) - Number(keys.has("ShiftLeft") || keys.has("ShiftRight"));
    const move = [axis("KeyD", "KeyA"), vertical, axis("KeyW", "KeyS")];
    if (!move.some(Boolean) && !look.some(Boolean)) return;
    fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ move, look, dt: 1 / 60 }),
    }).catch(console.error);
    look = [0, 0];
  }, 1000 / 60);
};
