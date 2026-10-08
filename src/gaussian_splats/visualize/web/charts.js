export function createCharts(hud) {
  const width = () => Math.max(100, hud.clientWidth - 24);
  const options = series => ({
    width: width(), height: 120,
    scales: { x: { time: false } },
    legend: { show: false },
    axes: [
      { stroke: "#9da7b3", grid: { stroke: "#333842" } },
      { stroke: "#9da7b3", grid: { stroke: "#333842" } },
    ],
    series: [{ label: "" }, ...series],
  });
  const fpsData = [[], []];
  const fps = new uPlot(options([{ label: "FPS", stroke: "#7ee787" }]),
                        fpsData, hud.querySelector(".fps-chart"));

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
  });
  observer.observe(hud);

  return {
    add(fpsValue) {
      append(fps, fpsData, [fpsValue]);
    },
    dispose() {
      observer.disconnect();
      fps.destroy();
    },
  };
}
