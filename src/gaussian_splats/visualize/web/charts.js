const LIMIT = 120;
const COLORS = ["#58a6ff", "#7ee787", "#d29922", "#a371f7", "#39c5cf", "#ff7b72"];

const plotOptions = (width, series, bands = []) => ({
  width,
  height: 120,
  scales: { x: { time: false } },
  axes: [
    { stroke: "#9da7b3", grid: { stroke: "#333842" } },
    { stroke: "#9da7b3", grid: { stroke: "#333842" } },
  ],
  legend: { show: true },
  bands,
  series,
});

const trim = data => {
  if (data[0].length > LIMIT) data.forEach(series => series.shift());
};

export function createCharts(elements, width) {
  if (typeof uPlot === "undefined") {
    return { add() {}, resize() {} };
  }
  const fpsData = [[], []];
  const fps = new uPlot(
    plotOptions(width, [{}, { label: "FPS", stroke: "#7ee787", value: (_u, value) => `${value?.toFixed(1) ?? "--"} FPS` }]),
    fpsData,
    elements.fps,
  );
  let timing;
  let timingData;
  let timingNames = [];

  return {
    add(fpsValue, timings) {
      const time = performance.now() / 1000;
      fpsData[0].push(time);
      fpsData[1].push(fpsValue);
      trim(fpsData);
      fps.setData(fpsData);

      if (!elements.timing || !timings) return;
      const names = Object.keys(timings);
      if (names.join() !== timingNames.join()) {
        timing?.destroy();
        timingNames = names;
        timingData = [[], ...names.map(() => [])];
        timing = new uPlot(
          plotOptions(
            width,
            [{}, ...names.map((name, index) => ({
              label: name,
              stroke: COLORS[index % COLORS.length],
              fill: `${COLORS[index % COLORS.length]}99`,
              value: (u, value, series, point) => {
                if (value == null) return "--";
                const lower = series > 1 ? u.data[series - 1][point] : 0;
                return `${(value - lower).toFixed(1)} ms`;
              },
            }))],
            names.map((_, index) => ({ series: [index + 1, index + 2] })),
          ),
          timingData,
          elements.timing,
        );
      }

      timingData[0].push(time);
      names.forEach((name, index) => {
        const previous = index ? timingData[index][timingData[index].length - 1] : 0;
        timingData[index + 1].push(previous + timings[name]);
      });
      trim(timingData);
      timing.setData(timingData);
      elements.total.textContent = `Total: ${names.reduce((sum, name) => sum + timings[name], 0).toFixed(1)} ms`;
    },
    resize(nextWidth) {
      width = nextWidth;
      fps.setSize({ width, height: 120 });
      timing?.setSize({ width, height: 120 });
    },
  };
}
