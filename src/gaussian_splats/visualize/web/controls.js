export function bindControls(canvas, { endpoint, hud }) {
  const keys = new Set();
  let look = [0, 0];
  let pending = false;

  canvas.tabIndex = 0;
  canvas.addEventListener("click", () => canvas.requestPointerLock());
  window.addEventListener("keydown", event => {
    if (["KeyW", "KeyA", "KeyS", "KeyD", "Space", "ShiftLeft", "ShiftRight"].includes(event.code)) {
      event.preventDefault();
      keys.add(event.code);
    }
    if (event.code === "KeyH" && !event.repeat && hud) hud.open = !hud.open;
  });
  window.addEventListener("keyup", event => keys.delete(event.code));
  window.addEventListener("mousemove", event => {
    if (document.pointerLockElement === canvas) {
      look[0] += event.movementX;
      look[1] += event.movementY;
    }
  });

  setInterval(async () => {
    const move = [
      Number(keys.has("KeyD")) - Number(keys.has("KeyA")),
      Number(keys.has("Space")) - Number(keys.has("ShiftLeft") || keys.has("ShiftRight")),
      Number(keys.has("KeyW")) - Number(keys.has("KeyS")),
    ];
    if (pending || (!move.some(Boolean) && !look.some(Boolean))) return;

    const currentLook = look;
    look = [0, 0];
    pending = true;
    try {
      await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ move, look: currentLook, dt: 1 / 20 }),
      });
    } finally {
      pending = false;
    }
  }, 50);
}
