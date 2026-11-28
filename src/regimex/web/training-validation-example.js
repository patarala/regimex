import {element} from "./ui.js";

const labels = {
  growth: "Growth", decline: "Decline", sideways: "Sideways",
  no_trades: "No trades", insufficient_data: "Insufficient data",
};
const colors = {
  growth: "var(--success)", decline: "var(--danger)", sideways: "var(--focus)",
  no_trades: "var(--muted)", insufficient_data: "var(--border)",
};

function svgNode(tag, attributes) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  return node;
}

function layer(length, height, sample) {
  const chart = svgNode("svg", {
    viewBox: `0 0 ${length} ${height}`, preserveAspectRatio: "none",
    height, width: "100%", "aria-hidden": true,
  });
  chart.append(svgNode("rect", {
    x: sample.missed_start - sample.window_start, width: sample.missed_stop - sample.missed_start,
    y: 0, height, fill: "var(--hover)",
  }));
  return chart;
}

function priceLine(sample, description) {
  const height = 76;
  const chart = layer(sample.prices.length, height, sample);
  const values = sample.prices.filter(Number.isFinite);
  if (!values.length) {
    description.textContent = "No observed prices in this window";
    return chart;
  }
  const low = Math.min(...values), high = Math.max(...values);
  description.textContent = `Last price · ${low}${low === high ? "" : " – " + high} ${sample.price_unit}`;
  const points = [];
  sample.prices.forEach((price, index) => {
    if (!Number.isFinite(price)) return;
    const x = index + 0.5;
    const y = high === low ? height / 2 : 4 + (high - price) / (high - low) * (height - 8);
    // A zero-length round stroke keeps an isolated observation visible across a gap.
    points.push(Number.isFinite(sample.prices[index - 1]) ? `L${x} ${y}` : `M${x} ${y}l0 0`);
  });
  chart.append(svgNode("path", {
    d: points.join(" "), fill: "none", stroke: "currentColor",
    "stroke-width": 2, "stroke-linecap": "round", "vector-effect": "non-scaling-stroke",
  }));
  return chart;
}

function intervalTrack(spans, sample) {
  const chart = layer(sample.prices.length, 14, sample);
  const paths = new Map();
  const end = sample.window_start + sample.prices.length;
  for (const span of spans) {
    const start = Math.max(span.start, sample.window_start);
    const stop = Math.min(span.stop, end);
    const rect = `M${start - sample.window_start} 2h${stop - start}v10H${start - sample.window_start}Z`;
    if (!paths.has(span.label)) paths.set(span.label, []);
    paths.get(span.label).push(rect);
  }
  for (const [label, rects] of paths)
    chart.append(svgNode("path", {d: rects.join(" "), fill: colors[label] || "var(--muted)"}));
  return chart;
}

export function validationExample(direction, sample, kind) {
  const figure = element("figure", undefined, "validation-example");
  const start = sample.window_start, stop = start + sample.prices.length;
  const range = span => `[${span.start}, ${span.stop})`;
  const caption = element("figcaption",
    `${sample.instrument_id} · Missed ${direction} interval [${sample.missed_start}, ${sample.missed_stop})`);
  const description = element("small", "", "hint");
  const grid = element("div", undefined, "validation-example-grid");
  const reference = kind === "human" ? "Human labels" : "Auto labels";
  grid.append(element("span", "Price"), priceLine(sample, description),
    element("span", reference), intervalTrack(sample.reference, sample),
    element("span", "Prediction"), intervalTrack(sample.prediction, sample));
  const axis = element("div", undefined, "validation-example-axis");
  axis.append(element("span", String(start)), element("span", "Step"), element("span", String(stop)));
  grid.append(element("span"), axis);
  const legend = element("div", undefined, "validation-example-legend");
  for (const label of new Set([...sample.reference, ...sample.prediction].map(span => span.label))) {
    const item = element("span");
    const swatch = svgNode("svg", {width: 10, height: 10, "aria-hidden": true});
    swatch.append(svgNode("rect", {width: 10, height: 10, fill: colors[label] || "var(--muted)"}));
    item.append(swatch, document.createTextNode(labels[label] || label));
    legend.append(item);
  }
  const details = element("details");
  details.append(element("summary", "Interval boundaries"));
  for (const [title, spans] of [[reference, sample.reference], ["Prediction", sample.prediction]]) {
    details.append(element("p", title + " · " + spans.map(span =>
      (labels[span.label] || span.label) + " " + range(span)).join(", ")));
  }
  details.append(element("p", "Experiment · " + sample.experiment_id, "hint"));
  figure.append(caption, description, grid, legend, details);
  return figure;
}
