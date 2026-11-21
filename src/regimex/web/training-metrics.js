import {$, option} from "./ui.js";

function node(tag, text) {
  const item = document.createElement(tag);
  item.textContent = text;
  return item;
}

function svgNode(tag, attrs, label = "") {
  const item = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) item.setAttribute(key, String(value));
  item.textContent = label;
  return item;
}

const format = (value) => value.toLocaleString(undefined, {maximumFractionDigits: 4});

// Keep local extrema visible while bounding the number of vertices drawn on a phone.
function envelope(points, width = 160) {
  const step = Math.ceil(points.length / width);
  if (step <= 2) return points;
  const reduced = [points[0]];
  for (let i = 0; i < points.length; i += step) {
    const chunk = points.slice(i, i + step);
    const low = chunk.reduce((a, b) => a[1] <= b[1] ? a : b);
    const high = chunk.reduce((a, b) => a[1] >= b[1] ? a : b);
    reduced.push(...(low[0] < high[0] ? [low, high] : [high, low]));
  }
  reduced.push(points[points.length - 1]);
  return reduced;
}

function plot(title, axis, points, unitScale) {
  const figure = document.createElement("figure");
  figure.append(node("figcaption", title));
  const graph = svgNode("svg", {viewBox: "0 0 550 151", role: "img",
    "aria-label": title + " by " + axis.toLowerCase()});
  const first = points[0][0], last = points[points.length - 1][0];
  const top = unitScale ? 1 : points.reduce((peak, p) => Math.max(peak, p[1]), 0) || 1;
  const x = (v) => last === first ? 283 : 49 + 475 * (v - first) / (last - first);
  const y = (v) => 111 - 95 * v / top;
  for (const value of [0, top / 2, top]) {
    graph.append(svgNode("line", {x1: 49, x2: 524, y1: y(value), y2: y(value), class: "training-grid"}),
      svgNode("text", {x: 42, y: y(value) + 4, "text-anchor": "end"}, format(value)));
  }
  graph.append(svgNode("polyline", {points: envelope(points).map(([a,b]) => x(a) + "," + y(b)).join(" "),
    class: unitScale ? "training-f1" : "training-loss"}));
  if (points.length <= 30)
    for (const [a, b] of points) {
      const point = svgNode("circle", {cx: x(a), cy: y(b), r: 3, class: "training-point"});
      point.append(svgNode("title", {}, axis + " " + format(a) + ": " + format(b)));
      graph.append(point);
    }
  graph.append(svgNode("text", {x: x(first), y: 129, "text-anchor": "middle"}, format(first)));
  if (last !== first) graph.append(svgNode("text", {x: x(last), y: 129, "text-anchor": "middle"}, format(last)));
  graph.append(svgNode("text", {x: 283, y: 148, "text-anchor": "middle"}, axis));
  figure.append(graph);
  return figure;
}

export function showTrainingMetrics(metrics) {
  $("training-metrics").hidden = false;
  $("training-score").textContent = "Selected validation interval macro-F1: " +
    format(metrics.validation_segment_macro_f1) + " · Seed " + metrics.selected_seed;
  const history = metrics.history || {};
  const schema = [["epochs", "epoch", "Epoch"], ["updates", "update", "Update"],
    ["candidates", "regularization", "Regularization"]].find(([name]) => Array.isArray(history[name]));
  const seed = $("training-seed");
  seed.replaceChildren();
  $("training-seed-label").hidden = true;
  $("training-charts").replaceChildren();
  seed.onchange = null;
  if (!schema) {
    $("training-charts").append(node("p", metrics.history_note || "This backend provides no plottable history."));
    return;
  }
  const [key, coordinate, axis] = schema;
  const rows = history[key].filter((row) => row && Number.isFinite(row[coordinate]));
  const seeds = [...new Set(rows.map((row) => row.seed).filter(Number.isInteger))];
  for (const value of seeds) seed.append(option(String(value), String(value)));
  seed.value = String(seeds.includes(metrics.selected_seed) ? metrics.selected_seed : seeds[0]);
  $("training-seed-label").hidden = seeds.length < 2;
  function draw() {
    const selected = rows.filter((row) => !seeds.length || row.seed === Number(seed.value))
      .sort((a, b) => a[coordinate] - b[coordinate]);
    const charts = $("training-charts");
    charts.replaceChildren();
    for (const [field, label] of [["training_loss", "Training loss"],
      ["validation_segment_macro_f1", "Validation interval macro-F1"]]) {
      const points = selected.filter((r) => Number.isFinite(r[field])).map((r) => [r[coordinate], r[field]]);
      if (points.length) charts.append(plot(label, axis, points, field !== "training_loss"));
    }
  }
  seed.onchange = draw;
  draw();
}
