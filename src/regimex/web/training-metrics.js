import {$, option} from "./ui.js";

function node(tag, text = "") {
  const element = document.createElement(tag);
  element.textContent = text;
  return element;
}

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
  const figure = node("figure");
  figure.append(node("figcaption", metric.label));
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
  figure.append(graph);
  return figure;
}

function valuesTable(history, records, metrics, open) {
  const details = node("details");
  details.className = "training-observations";
  details.open = open;
  details.append(node("summary", "Metric values"));
  const table = node("table");
  table.setAttribute("aria-label", "Recorded metric values by " + history.label.toLowerCase());
  const headings = table.createTHead().insertRow();
  for (const label of [history.label, ...metrics.map(metric => metric.label)]) {
    const cell = node("th", label);
    cell.scope = "col";
    headings.append(cell);
  }
  const body = table.createTBody();
  const wrapper = node("div");
  wrapper.className = "training-values";
  wrapper.append(table);
  details.append(wrapper);
  const perPage = 20;
  const previous = node("button", "Previous values"), next = node("button", "Next values");
  const range = node("span");
  range.setAttribute("role", "status");
  let offset = 0;
  function render() {
    body.replaceChildren();
    for (const record of records.slice(offset, offset + perPage)) {
      const row = body.insertRow();
      const index = node("th", format(record[history.field]));
      index.scope = "row";
      row.append(index);
      for (const {field} of metrics) {
        row.insertCell().textContent = Number.isFinite(record[field]) ? format(record[field]) : "Not recorded";
      }
    }
    const limit = Math.min(offset + perPage, records.length);
    previous.disabled = offset === 0;
    next.disabled = limit === records.length;
    range.textContent = (offset + 1) + "–" + limit + " of " + records.length;
  }
  if (records.length > perPage) {
    const nav = node("div");
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
  }
  render();
  return details;
}

export function showTrainingMetrics(metrics) {
  $("training-metrics").hidden = false;
  $("training-score").replaceChildren("The saved model (seed " + metrics.selected_seed + ") scored ",
    node("strong", format(metrics.validation_segment_macro_f1)), " on validation interval macro-F1.");
  const charts = $("training-charts"), seed = $("training-seed");
  charts.replaceChildren();
  seed.replaceChildren();
  $("training-seed-label").hidden = true;
  seed.onchange = null;
  const history = histories.find(schema => Array.isArray(metrics.history?.[schema.rows]));
  const records = history && metrics.history[history.rows].filter(row => row && Number.isFinite(row[history.field]));
  if (!records?.length) {
    charts.append(node("p", metrics.history_note || "The run has no recorded metric values to plot."));
    return;
  }
  const seeds = [...new Set(records.map(row => row.seed).filter(Number.isInteger))];
  for (const value of seeds) seed.append(option(String(value), String(value)));
  if (seeds.length) seed.value = String(seeds.includes(metrics.selected_seed) ? metrics.selected_seed : seeds[0]);
  $("training-seed-label").hidden = seeds.length < 2;
  function draw() {
    const keepTableOpen = charts.querySelector("details")?.open || false;
    const selected = records.filter(row => !seeds.length || row.seed === Number(seed.value))
      .sort((a, b) => a[history.field] - b[history.field]);
    const axis = horizontalAxis(history, selected);
    const series = [];
    for (const metric of metricsToPlot) {
      const points = selected.filter(row => Number.isFinite(row[metric.field]))
        .map(row => [row[history.field], row[metric.field]]);
      if (!points.length) continue;
      const ceiling = metric.field === "training_loss" ? lossCeiling(points) : 1;
      const ticks = [0, ceiling / 2, ceiling].map(value => ({value, label: format(value)}));
      series.push({...metric, points, ceiling, ticks});
    }
    if (!series.length) {
      charts.replaceChildren(node("p", "No metric values were recorded for this seed."));
      return;
    }
    const tickWidths = series.flatMap(chart => chart.ticks.map(({label}) => label.length * 9));
    const gutter = Math.max(36, Math.max(...tickWidths) + 14);
    charts.replaceChildren(...series.map(metric => plot(metric, axis, gutter)),
      valuesTable(history, selected, series, keepTableOpen));
  }
  seed.onchange = draw;
  draw();
}
