const LIMIT = 120;
const COLORS = [
  "#58a6ff", "#7ee787", "#d29922", "#a371f7", "#39c5cf",
  "#f2cc60", "#ff7b72", "#db61a2", "#f85149",
];

const options = (width, scales = {}) => ({
  width, height: 120,
  scales: { x: { time: false }, ...scales },
  legend: { show: true },
  axes: Array.from({ length: 2 }, () => ({ stroke: "#fff", grid: { stroke: "#333" } })),
});

const append = (data, values) => {
  data.forEach((series, index) => series.push(values[index]));
  if (data[0].length > LIMIT) data.forEach(series => series.shift());
};

const format = (unit, stacked = false) => (u, _value, series, index) => {
  index ??= u.data[0].length - 1;
  if (index < 0) return "--";
  const previous = stacked && series > 1 ? u.data[series - 1][index] : 0;
  return `${(u.data[series][index] - previous).toFixed(1)} ${unit}`;
};

const stack = data => {
  const totals = Array(data[0].length).fill(0);
  return [data[0], ...data.slice(1).map(series => series.map((value, i) => totals[i] += value))];
};

export const createCharts = (elements, width) => {
  const started = performance.now();
  const fpsData = [[], []];
  const fps = new uPlot({
    ...options(width),
    series: [{}, { label: "FPS", stroke: "#7ee787", value: format("FPS") }],
  }, fpsData, elements.fps);

  let timing;
  let timingData = [[]];
  let stages = [];

  const configure = times => {
    const names = Object.keys(times);
    if (!names.length) return false;
    if (names.join() === stages.join()) return true;
    timing?.destroy();
    stages = names;
    timingData = [[], ...stages.map(() => [])];
    timing = new uPlot({
      ...options(width, { y: { range: (_u, _min, max) => [0, uPlot.rangeNum(0, max, 0.1, true)[1]] } }),
      bands: stages.map((_, i) => ({ series: [i === stages.length - 1 ? -1 : i + 2, i + 1] })),
      series: [{}, ...stages.map((name, i) => ({
        label: name, value: format("ms", true),
        stroke: COLORS[i % COLORS.length], fill: `${COLORS[i % COLORS.length]}99`,
      }))],
    }, timingData, elements.timing);
    return true;
  };

  return {
    add(fpsValue, times) {
      const elapsed = (performance.now() - started) / 1000;
      append(fpsData, [elapsed, fpsValue]);
      fps.setData(fpsData);
      if (!times || !configure(times)) return;
      append(timingData, [elapsed, ...stages.map(stage => times[stage])]);
      timing.setData(stack(timingData));
      const total = Object.values(times).reduce((sum, value) => sum + value, 0);
      elements.total.textContent = `Total: ${total.toFixed(1)} ms`;
    },
    resize(nextWidth) {
      width = nextWidth;
      fps.setSize({ width, height: 120 });
      timing?.setSize({ width, height: 120 });
    },
  };
};
