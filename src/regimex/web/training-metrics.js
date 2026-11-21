import {$, element, option} from "./ui.js";

function svgNode(tag, attrs, text = "") {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
  element.textContent = text;
  return element;
}

const decimal = new Intl.NumberFormat(undefined, {maximumSignificantDigits: 5});
const scientific = new Intl.NumberFormat(undefined, {notation: "scientific", maximumSignificantDigits: 3});

function format(value) {
  const magnitude = Math.abs(value);
  return (magnitude > 0 && (magnitude < 0.001 || magnitude >= 1e6) ? scientific : decimal).format(value);
}

const histories = [
  {rows: "epochs", field: "epoch", label: "Epoch"},
  {rows: "updates", field: "update", label: "Update"},
  {rows: "candidates", field: "regularization", label: "Regularization"},
];
const metricsToPlot = [
  {field: "training_loss", label: "Training loss", style: "training-loss"},
  {field: "validation_segment_macro_f1", label: "Validation interval macro-F1", style: "training-f1"},
];

// Retain local minima and maxima when thousands of updates share a few hundred pixels.
function envelope(points, buckets = 160) {
  const step = Math.ceil(points.length / buckets);
  if (step <= 2) return points;
  const reduced = [points[0]];
  for (let i = 0; i < points.length; i += step) {
    const chunk = points.slice(i, i + step);
    const low = chunk.reduce((a, b) => a[1] <= b[1] ? a : b);
    const high = chunk.reduce((a, b) => a[1] >= b[1] ? a : b);
    reduced.push(...(low[0] < high[0] ? [low, high] : [high, low]));
  }
  reduced.push(points.at(-1));
  return reduced;
}

function horizontalAxis(history, records) {
  const values = records.map(row => row[history.field]);
  const first = values[0], last = values.at(-1);
  const useLog = history.field === "regularization" && first > 0 && last / first >= 100;
  const scale = useLog ? Math.log : value => value;
  const start = scale(first), span = scale(last) - start;
  const percent = value => span ? 100 * (scale(value) - start) / span : 50;
  const middle = values.reduce((a, b) => Math.abs(percent(a) - 50) <= Math.abs(percent(b) - 50) ? a : b);
  const ticks = first === last ? [first] : [first, last];
  if (first !== last && percent(middle) >= 35 && percent(middle) <= 65) ticks.splice(1, 0, middle);
  return {label: history.label + (useLog ? " (log scale)" : ""), percent, ticks};
}

function lossCeiling(points) {
  const peak = points.reduce((max, [, value]) => Math.max(max, value), 0);
  if (!peak) return 1;
  const magnitude = 10 ** Math.floor(Math.log10(peak));
  const steps = [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10];
  const ceiling = magnitude * steps.find(step => magnitude * step >= peak);
  return Number.isFinite(ceiling) ? ceiling : peak;
}

function plot(metric, axis, gutter) {
  const {points, ceiling, ticks} = metric;
  const caption = element("figcaption", metric.label);
  if (metric.sampling) {
    const sampling = element("span", metric.sampling);
    sampling.className = "metric-sampling";
    caption.append(" ", sampling);
  }
  const graph = svgNode("svg", {width: "100%", height: 168, role: "img",
    "aria-label": metric.label + " by " + axis.label.toLowerCase()});
  const start = points[0], end = points.at(-1);
  const describe = ([x, y]) => axis.label + " " + format(x) + ": " + format(y);
  graph.append(svgNode("desc", {}, points.length === 1 ? describe(start) :
    describe(start) + ". " + describe(end) + "."));
  // Only the line stretches horizontally. Labels and point markers retain their size on phones.
  const height = 106;
  const area = svgNode("svg", {x: gutter, y: 12, width: "calc(100% - " + (gutter + 16) + "px)",
    height, overflow: "visible"});
  const y = value => height * (1 - value / ceiling);
  for (const {value, label} of ticks) {
    area.append(svgNode("line", {x1: 0, x2: "100%", y1: y(value), y2: y(value), class: "training-grid"}),
      svgNode("text", {x: -8, y: y(value) + 4, "text-anchor": "end"}, label));
  }
  const line = svgNode("svg", {width: "100%", height, viewBox: "0 0 100 " + height,
    preserveAspectRatio: "none", overflow: "visible"});
  line.append(svgNode("polyline", {points: envelope(points).map(([x, v]) => axis.percent(x) + "," + y(v)).join(" "),
    class: metric.style, "vector-effect": "non-scaling-stroke"}));
  area.append(line);
  if (points.length <= 30) {
    for (const point of points) {
      const marker = svgNode("circle", {cx: axis.percent(point[0]) + "%", cy: y(point[1]), r: 3,
        class: "training-point"});
      marker.append(svgNode("title", {}, describe(point)));
      area.append(marker);
    }
  }
  for (const tick of axis.ticks) {
    const x = axis.percent(tick);
    area.append(svgNode("text", {x: x + "%", y: height + 20,
      "text-anchor": x === 0 ? "start" : x === 100 ? "end" : "middle"}, format(tick)));
  }
  area.append(svgNode("text", {x: "50%", y: height + 42, "text-anchor": "middle"}, axis.label));
  graph.append(area);
  return [caption, graph];
}

function valuesTable(history, metric) {
  let points = [], offset = 0;
  const details = element("details");
  details.className = "training-observations";
  details.dataset.metric = metric.field;
  const summary = element("summary", "Values");
  summary.setAttribute("aria-label", metric.label + " values");
  details.append(summary);
  const table = element("table");
  table.setAttribute("aria-label", metric.label + " by " + history.label.toLowerCase());
  const headings = table.createTHead().insertRow();
  for (const label of [history.label, metric.label]) {
    const cell = element("th", label);
    cell.scope = "col";
    headings.append(cell);
  }
  const body = table.createTBody();
  const wrapper = element("div");
  wrapper.className = "training-values";
  wrapper.append(table);
  details.append(wrapper);
  const perPage = 20;
  const previous = element("button", "Previous values"), next = element("button", "Next values");
  const range = element("span");
  range.setAttribute("role", "status");
  function render() {
    body.replaceChildren();
    for (const [x, value] of points.slice(offset, offset + perPage)) {
      const row = body.insertRow();
      const index = element("th", format(x));
      index.scope = "row";
      row.append(index);
      row.insertCell().textContent = format(value);
    }
    const limit = Math.min(offset + perPage, points.length);
    previous.disabled = offset === 0;
    next.disabled = limit === points.length;
    range.textContent = (offset + 1) + "–" + limit + " of " + points.length;
  }
  const nav = element("nav");
  nav.setAttribute("aria-label", metric.label + " values");
  nav.className = "training-values-nav";
  for (const [button, delta] of [[previous, -perPage], [next, perPage]]) {
    button.type = "button";
    button.className = "quiet";
    button.addEventListener("click", () => {
      offset += delta;
      render();
      if (button.disabled) (button === next ? previous : next).focus();
    });
  }
  nav.append(previous, range, next);
  details.append(nav);
  function update(values, resetPage) {
    points = values;
    if (resetPage || offset >= points.length) offset = 0;
    nav.hidden = points.length <= perPage;
    render();
  }
  return {details, update};
}

function chartSeries(history, selected) {
  const series = [];
  for (const metric of metricsToPlot) {
    const points = selected.filter(row => Number.isFinite(row[metric.field]))
      .map(row => [row[history.field], row[metric.field]]);
    if (!points.length) continue;
    const ceiling = metric.field === "training_loss" ? lossCeiling(points) : 1;
    const ticks = [0, ceiling / 2, ceiling].map(value => ({value, label: format(value)}));
    let sampling = "";
    if (metric.field === "validation_segment_macro_f1" && points.length < selected.length &&
        (history.field === "epoch" || history.field === "update")) {
      sampling = points.length + " of " + selected.length + " " + history.field + "s evaluated";
    }
    series.push({...metric, points, ceiling, ticks, sampling});
  }
  return series;
}

export function createTrainingMetrics() {
  const section = $("training-metrics"), charts = $("training-charts"), seed = $("training-seed");
  const score = $("training-score"), seedLabel = $("training-seed-label");
  const figures = new Map();
  let chosenSeed = "", drawn = null;

  function clear() {
    section.hidden = true;
    seedLabel.hidden = score.hidden = true;
    score.replaceChildren();
    chosenSeed = "";
    drawn = null;
    figures.clear();
    charts.replaceChildren();
    seed.replaceChildren();
    seed.onchange = null;
  }

  function show(historyData, result = null) {
    const history = histories.find(schema => Array.isArray(historyData?.[schema.rows]));
    const rows = history && historyData[history.rows];
    const records = rows?.filter(row => row && Number.isFinite(row[history.field]));
    section.hidden = false;
    score.hidden = !result;
    if (result) {
      const saved = history?.rows === "candidates" ? "saved model" : "saved model (seed " + result.selected_seed + ")";
      score.replaceChildren(element("strong", format(result.validation_segment_macro_f1)),
        " validation interval macro-F1 · " + saved);
    }
    if (!records?.length) {
      figures.clear();
      charts.replaceChildren(element("p", result?.history_note || "The run has no recorded metric values to plot."));
      seedLabel.hidden = true;
      return;
    }
    const seeds = [...new Set(records.map(row => row.seed).filter(Number.isInteger))];
    const savedSeed = result?.selected_seed;
    for (const [index, value] of seeds.entries()) {
      const label = "Seed " + value + (value === savedSeed ? " (saved model)" : "");
      const item = seed.options[index];
      if (!item) seed.append(option(String(value), label));
      else if (item.value !== String(value) || item.text !== label) item.replaceWith(option(String(value), label));
    }
    while (seed.options.length > seeds.length) seed.lastElementChild.remove();
    // Follow the repeat being trained until the reader explicitly chooses one.
    if (seeds.length) {
      const automatic = seeds.includes(savedSeed) ? savedSeed : seeds.at(-1);
      seed.value = chosenSeed && seeds.includes(Number(chosenSeed)) ? chosenSeed : String(automatic);
    }
    seedLabel.hidden = seeds.length < 2;
    function draw() {
      const selected = records.filter(row => !seeds.length || row.seed === Number(seed.value));
      if (drawn?.rows === rows && drawn.seed === seed.value && drawn.count === selected.length) return;
      const resetPage = drawn?.seed !== seed.value;
      drawn = {rows, seed: seed.value, count: selected.length};
      selected.sort((a, b) => a[history.field] - b[history.field]);
      const axis = horizontalAxis(history, selected);
      const series = chartSeries(history, selected);
      for (const [key, view] of figures) {
        if (!series.some(metric => metric.field === key)) {
          view.figure.remove();
          figures.delete(key);
        }
      }
      if (!series.length) {
        charts.replaceChildren(element("p", "No metric values were recorded for this seed."));
        return;
      }
      if (!figures.size) charts.replaceChildren();
      const tickWidths = series.flatMap(chart => chart.ticks.map(({label}) => label.length * 9));
      const gutter = Math.max(36, Math.max(...tickWidths) + 14);
      for (const [index, metric] of series.entries()) {
        let view = figures.get(metric.field);
        const [caption, graph] = plot(metric, axis, gutter);
        if (!view) {
          const figure = element("figure"), values = valuesTable(history, metric);
          figure.append(caption, graph, values.details);
          charts.insertBefore(figure, charts.children[index] || null);
          view = {figure, values};
          figures.set(metric.field, view);
        } else {
          view.figure.querySelector("figcaption").replaceWith(caption);
          view.figure.querySelector("svg").replaceWith(graph);
        }
        view.values.update(metric.points, resetPage);
      }
    }
    seed.onchange = () => { chosenSeed = seed.value; draw(); };
    draw();
  }
  return {show, clear};
}
