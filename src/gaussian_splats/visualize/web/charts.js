const COLORS = ["#58a6ff", "#7ee787", "#d29922", "#a371f7", "#39c5cf", "#ff7b72"];

export function createCharts(hud) {
  const width = () => Math.max(100, hud.clientWidth - 24);
  const options = series => ({
    width: width(), height: 120,
    scales: { x: { time: false } },
    legend: { live: true },
    axes: [
      { stroke: "#9da7b3", grid: { stroke: "#333842" } },
      { stroke: "#9da7b3", grid: { stroke: "#333842" } },
    ],
    series: [{ label: "" }, ...series],
  });
  const fpsData = [[], []];
  const fps = new uPlot(options([{ label: "FPS", stroke: "#7ee787" }]),
                        fpsData, hud.querySelector(".fps-chart"));
  let timing, timingData, names = [];

  function append(plot, data, values) {
    data[0].push(performance.now() / 1000);
    values.forEach((value, index) => data[index + 1].push(value));
    if (data[0].length > 120) data.forEach(series => series.shift());
    plot.setData(data);
    plot.setCursor({ idx: data[0].length - 1 });
  }

  const observer = new ResizeObserver(() => {
    if (!hud.open) return;
    fps.setSize({ width: width(), height: 120 });
    timing?.setSize({ width: width(), height: 120 });
  });
  observer.observe(hud);

  return {
    add(fpsValue, timings) {
      append(fps, fpsData, [fpsValue]);
      const next = Object.keys(timings);
      if (JSON.stringify(next) !== JSON.stringify(names)) {
        timing?.destroy();
        timing = undefined;
        names = next;
        timingData = [[], ...names.map(() => [])];
        if (names.length) {
          timing = new uPlot(options(names.map((name, index) => ({
            label: name, stroke: COLORS[index % COLORS.length],
            value: (_plot, value) => value == null ? "--" : `${value.toFixed(1)} ms`,
          }))), timingData, hud.querySelector(".timing-chart"));
        }
      }
      if (timing) append(timing, timingData, names.map(name => timings[name]));
    },
    dispose() {
      observer.disconnect();
      fps.destroy();
      timing?.destroy();
    },
  };
}
