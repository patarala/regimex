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
  {rows: "fits", field: "trees", label: "Trees used"},
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
  const values = records.map(row => row[history.field]).sort((a, b) => a - b);
  const first = values[0], last = values.at(-1);
  const useLog = history.field === "regularization" && first > 0 && last / first >= 100;
  const scale = useLog ? Math.log : value => value;
  const start = scale(first), span = scale(last) - start;
  const percent = value => span ? 100 * (scale(value) - start) / span : 50;
  const middle = values.reduce((a, b) => Math.abs(percent(a) - 50) <= Math.abs(percent(b) - 50) ? a : b);
  const ticks = first === last ? [first] : [first, last];
  if (first !== last && percent(middle) >= 35 && percent(middle) <= 65) ticks.splice(1, 0, middle);
  const reading = history.field === "trees" ? treeCount : value => history.label + " " + format(value);
  return {dimension: history.label, label: history.label + (useLog ? " (log scale)" : ""), reading, percent, ticks};
}

function treeCount(value) {
  return format(value) + (value === 1 ? " tree" : " trees");
}

function lossCeiling(points) {
  const peak = points.reduce((max, [, value]) => Math.max(max, value), 0);
  if (!peak) return 1;
  const magnitude = 10 ** Math.floor(Math.log10(peak));
  const steps = [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10];
  const ceiling = magnitude * steps.find(step => magnitude * step >= peak);
  return Number.isFinite(ceiling) ? ceiling : peak;
}

const plotHeight = 106, plotRight = 16, plotTop = 12;

function plotY(metric, value) {
  return plotHeight * ((metric.ceiling - value) / (metric.ceiling - metric.floor));
}

function plot(metric, axis, gutter) {
  const {points, ticks} = metric;
  const caption = element("figcaption", metric.label);
  if (metric.sampling) {
    const sampling = element("span", metric.sampling);
    sampling.className = "metric-sampling";
    caption.append(" ", sampling);
  }
  const graph = svgNode("svg", {width: "100%", height: 168, role: "img",
    "aria-label": metric.label + " by " + axis.label.toLowerCase()});
  const start = points[0], end = points.at(-1);
  const describe = ([x, y]) => axis.reading(x) + ": " + format(y);
  const range = start === end ? describe(end) : describe(start) + ". " + describe(end);
  graph.append(svgNode("desc", {}, range + "." + (metric.best ? " Best validation score: " + describe(metric.best) + "." : "")));
  // Only the line stretches horizontally. Labels and point markers retain their size on phones.
  const height = plotHeight;
  const area = svgNode("svg", {x: gutter, y: plotTop, width: "calc(100% - " + (gutter + plotRight) + "px)",
    height, overflow: "visible"});
  const y = value => plotY(metric, value);
  for (const {value, label} of ticks) {
    area.append(svgNode("line", {x1: 0, x2: "100%", y1: y(value), y2: y(value), class: "training-grid"}),
      svgNode("text", {x: -8, y: y(value) + 4, "text-anchor": "end"}, label));
  }
  const line = svgNode("svg", {width: "100%", height, viewBox: "0 0 100 " + height,
    preserveAspectRatio: "none", overflow: "visible"});
  line.append(svgNode("polyline", {points: envelope(points).map(([x, v]) => axis.percent(x) + "," + y(v)).join(" "),
    class: metric.style, "vector-effect": "non-scaling-stroke"}));
  area.append(line);
  const latest = points[metric.latestIndex];
  for (const point of metric.best && metric.best !== latest ? [metric.best, latest] : [latest]) {
    const marker = svgNode("circle", {cx: axis.percent(point[0]) + "%", cy: y(point[1]),
      r: point === metric.best ? 4.5 : 3.5, class: "training-point"});
    if (point === latest) marker.dataset.last = "";
    if (point === metric.best) marker.dataset.best = "";
    marker.append(svgNode("title", {}, describe(point)));
    area.append(marker);
  }
  for (const tick of axis.ticks) {
    const x = axis.percent(tick);
    area.append(svgNode("text", {x: x + "%", y: height + 20,
      "text-anchor": x === 0 ? "start" : x === 100 ? "end" : "middle"}, format(tick)));
  }
  area.append(svgNode("text", {x: "50%", y: height + 42, "text-anchor": "middle"}, axis.label));
  graph.append(area);
  return {caption, graph, area};
}

// Snap in screen coordinates so log-spaced candidates and unsampled intervals remain truthful.
function nearestPoint(points, percent, at) {
  let low = 0, high = points.length - 1;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (percent(points[mid][0]) < at) low = mid + 1;
    else high = mid;
  }
  if (low && at - percent(points[low - 1][0]) <= percent(points[low][0]) - at) return low - 1;
  return low;
}

function chartInspector() {
  const container = element("div", "", "training-chart");
  const control = element("div", "", "training-chart-inspector");
  control.tabIndex = 0;
  control.setAttribute("role", "slider");
  control.setAttribute("aria-description", "Use the arrow keys to move between recorded measurements.");
  const tooltip = element("div", "", "training-chart-tooltip");
  tooltip.setAttribute("role", "tooltip");
  tooltip.hidden = true;
  const reading = element("strong"), stepLabel = element("span");
  tooltip.append(reading, stepLabel);
  const crosshair = svgNode("line", {class: "training-crosshair", y1: 0, y2: plotHeight});
  const dot = svgNode("circle", {class: "training-hover-point", r: 4});
  container.append(control, tooltip);
  let graph, area, metric, axis, gutter, index = 0, mode, listeners;

  function hide() {
    mode = null;
    tooltip.hidden = true;
    crosshair.remove();
    dot.remove();
    listeners?.abort();
    listeners = null;
  }
  function updateReadout() {
    const [x, value, row] = metric.points[index];
    reading.textContent = format(value);
    stepLabel.textContent = axis.reading(x);
    const lines = [stepLabel.textContent, metric.label + ": " + reading.textContent];
    for (const other of metricsToPlot) {
      if (other.field !== metric.field && Number.isFinite(row[other.field]))
        lines.push(other.label + ": " + format(row[other.field]));
    }
    if (metric.points[index] === metric.best) lines.push("Best validation score");
    if (index === metric.latestIndex) lines.push("Latest measurement");
    control.setAttribute("aria-valuenow", x);
    control.setAttribute("aria-valuetext", lines.join(". "));
  }
  function place() {
    if (!mode) return;
    const rect = control.getBoundingClientRect(), width = rect.width - gutter - plotRight;
    const [x, value] = metric.points[index];
    const position = axis.percent(x) + "%", y = plotY(metric, value);
    crosshair.setAttribute("x1", position);
    crosshair.setAttribute("x2", position);
    dot.setAttribute("cx", position);
    dot.setAttribute("cy", y);
    const px = gutter + width * axis.percent(x) / 100;
    const box = tooltip.getBoundingClientRect();
    const fitsRight = px + 10 + box.width <= rect.width;
    const fitsLeft = px - box.width - 10 >= 0;
    const left = fitsRight ? px + 10 : px - box.width - 10;
    tooltip.style.left = Math.max(0, Math.min(left, rect.width - box.width)) + "px";
    let top = plotTop + y;
    if (!fitsLeft && !fitsRight) {
      const above = top - Math.max(0, -rect.top);
      const below = Math.min(rect.height, window.innerHeight - rect.top) - top;
      top += below >= box.height + 8 || below >= above ? 8 : -box.height - 8;
    }
    top = Math.max(0, Math.min(top, rect.height - box.height));
    tooltip.style.top = Math.max(-rect.top, Math.min(top, window.innerHeight - rect.top - box.height)) + "px";
  }
  function inspect(next, source) {
    if (next === index && mode) {
      mode = source;
      return;
    }
    if (next !== index) {
      index = next;
      updateReadout();
    }
    mode = source;
    tooltip.hidden = false;
    if (crosshair.parentNode !== area) area.append(crosshair, dot);
    place();
    if (!listeners) {
      listeners = new AbortController();
      const {signal} = listeners;
      document.addEventListener("pointerdown", event => {
        if (!control.contains(event.target)) hide();
      }, {capture: true, signal});
      document.addEventListener("keydown", event => {
        if (event.key === "Escape") hide();
      }, {signal});
      window.addEventListener("resize", place, {signal});
      const resize = new ResizeObserver(place);
      resize.observe(container);
      signal.addEventListener("abort", () => resize.disconnect(), {once: true});
    }
  }
  function locate(event) {
    const rect = control.getBoundingClientRect();
    const percent = 100 * (event.clientX - rect.left - gutter) / Math.max(1, rect.width - gutter - plotRight);
    inspect(nearestPoint(metric.points, axis.percent, percent), event.pointerType === "touch" ? "touch" : "mouse");
  }
  control.addEventListener("pointermove", locate);
  control.addEventListener("pointerenter", event => { if (event.pointerType !== "touch") locate(event); });
  control.addEventListener("pointerdown", locate);
  control.addEventListener("pointerleave", () => { if (mode === "mouse") hide(); });
  control.addEventListener("pointercancel", hide);
  control.addEventListener("focus", () => {
    if (control.matches(":focus-visible") && !mode) inspect(metric.latestIndex, "keyboard");
  });
  control.addEventListener("blur", hide);
  control.addEventListener("keydown", event => {
    if (event.key === "Escape") { hide(); return; }
    const last = metric.points.length - 1;
    const positions = {ArrowLeft: Math.max(0, index - 1), ArrowRight: Math.min(last, index + 1),
      ArrowDown: Math.max(0, index - 1), ArrowUp: Math.min(last, index + 1), Home: 0, End: last};
    if (!Object.hasOwn(positions, event.key)) return;
    event.preventDefault();
    inspect(positions[event.key], "keyboard");
  });

  function update(drawing, nextMetric, nextAxis, nextGutter, reset) {
    const previousX = metric?.points[index][0], previousMode = mode;
    hide();
    graph?.remove();
    ({graph, area} = drawing);
    container.prepend(graph);
    metric = nextMetric;
    axis = nextAxis;
    gutter = nextGutter;
    const points = metric.points;
    index = previousMode && !reset ? nearestPoint(points, axis.percent, axis.percent(previousX)) : metric.latestIndex;
    control.setAttribute("aria-label", "Inspect " + metric.label.toLowerCase() + " by " + axis.dimension.toLowerCase());
    control.setAttribute("aria-valuemin", points[0][0]);
    control.setAttribute("aria-valuemax", points.at(-1)[0]);
    updateReadout();
    if (previousMode && !reset) inspect(index, previousMode);
  }
  return {container, update, hide};
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

function chartSeries(history, selected, fromSavedWeights) {
  const series = [];
  for (const metric of metricsToPlot) {
    const points = selected.filter(row => Number.isFinite(row[metric.field]))
      .map(row => [row[history.field], row[metric.field], row]);
    if (!points.length) continue;
    const forest = history.field === "trees";
    let floor = 0, ceiling = metric.field === "training_loss" ? lossCeiling(points) : 1;
    let sampling = "";
    if (forest) {
      const low = points.reduce((min, [, score]) => Math.min(min, score), 1);
      const high = points.reduce((max, [, score]) => Math.max(max, score), 0);
      floor = Math.max(0, Math.floor(10 * (low - 0.01)) / 10);
      ceiling = Math.min(1, Math.ceil(10 * (high + 0.01)) / 10);
      sampling = fromSavedWeights ? "Measured from saved weights, adding trees in their original order."
        : "Each point adds trees from the same fitted forest.";
    }
    const ticks = [floor, (floor + ceiling) / 2, ceiling].map(value => ({value, label: format(value)}));
    if (metric.field === "validation_segment_macro_f1" && points.length < selected.length &&
        (history.field === "epoch" || history.field === "update")) {
      sampling = points.length + " of " + selected.length + " " + history.field + "s evaluated";
    }
    // Candidates can be evaluated in a different order from their numeric x-axis.
    const latest = points.at(-1);
    const best = metric.field === "validation_segment_macro_f1" && !forest ?
      points.reduce((first, point) => point[1] > first[1] ? point : first) : null;
    points.sort((a, b) => a[0] - b[0]);
    series.push({...metric, points, floor, ceiling, ticks, sampling, best, latestIndex: points.indexOf(latest)});
  }
  return series;
}

function repeatEnding(reason, step, policy) {
  if (reason === "early_stopping") {
    if (!policy?.epochs || !policy?.patience) return "Stopped early at epoch " + step + ".";
    const checks = policy.patience === 1 ? "1 validation check" : policy.patience + " consecutive validation checks";
    return `Stopped early at epoch ${step} of ${policy.epochs}. Interval macro-F1 did not improve for ${checks}.`;
  }
  if (reason === "epoch_limit") return step === 1 ? "Completed the scheduled epoch." : `Completed all ${step} epochs.`;
  if (reason === "update_limit") return step === 1 ? "Completed the scheduled optimizer update." : `Completed all ${step} optimizer updates.`;
  return "";
}

function directionQualityTable(rows) {
  const table = element("table", undefined, "evaluation-directions");
  table.createCaption().textContent = "Interval macro-F1 by reference";
  const splits = [["train", "Training"], ["validation", "Validation"], ["independent", "Independent"]]
    .filter(([split]) => rows.some(row => row.split === split));
  const header = table.createTHead().insertRow();
  for (const label of ["Reference", ...splits.map(([, label]) => label)]) {
    const heading = element("th", label);
    heading.scope = "col";
    header.append(heading);
  }
  const body = table.createTBody();
  for (const [reference, label] of [["simulator", "Simulator agreement"], ["known_direction", "Known direction"]]) {
    const measurements = rows.filter(row => row.reference === reference);
    if (!measurements.length) continue;
    const row = body.insertRow();
    const heading = element("th", label);
    heading.scope = "row";
    row.append(heading);
    for (const [split] of splits) {
      const cell = row.insertCell();
      const measurement = measurements.find(item => item.split === split);
      if (measurement) {
        const groups = measurement.group_count;
        cell.append(format(measurement.interval_macro_f1),
          element("small", groups.toLocaleString() + (groups === 1 ? " group" : " groups")));
      } else {
        cell.textContent = "—";
        cell.setAttribute("aria-label", "Not measured");
      }
    }
  }
  return table;
}

export function createTrainingMetrics() {
  const section = $("training-metrics"), charts = $("training-charts"), seed = $("training-seed");
  const score = $("training-score"), seedLabel = $("training-seed-label");
  const ending = $("training-repeat-end");
  const quality = $("training-direction-quality");
  const figures = new Map();
  let chosenSeed = "", drawn = null;

  function showQuality(report) {
    if (quality.dataset.report === report?.report_sha256) return;
    if (report) {
      quality.replaceChildren(directionQualityTable(report.rows));
      quality.dataset.report = report.report_sha256;
    } else {
      quality.replaceChildren();
      delete quality.dataset.report;
    }
    quality.hidden = !report;
  }

  function clear() {
    section.hidden = true;
    ending.hidden = seedLabel.hidden = score.hidden = true;
    score.replaceChildren();
    showQuality();
    chosenSeed = "";
    drawn = null;
    for (const view of figures.values()) view.inspector.hide();
    figures.clear();
    charts.replaceChildren();
    seed.replaceChildren();
    seed.onchange = null;
  }

  function show(historyData, result, policy) {
    const measured = result?.forest_validation_curve?.points;
    const fromSavedWeights = Array.isArray(measured) && measured.length > 1 && !(historyData?.fits?.length > 1);
    const data = fromSavedWeights ? {fits: measured} : historyData;
    const history = histories.find(schema => Array.isArray(data?.[schema.rows]));
    const rows = history && data[history.rows];
    const records = rows?.filter(row => row && Number.isFinite(row[history.field]));
    section.hidden = false;
    showQuality(result?.direction_quality);
    const fit = historyData?.fits?.at(-1);
    const validated = result || (fit && {selected_seed: fit.seed,
      validation_segment_macro_f1: fit.validation_segment_macro_f1});
    score.hidden = !validated;
    if (validated) {
      let selection = "seed " + validated.selected_seed;
      if (history?.rows === "candidates") selection = "selected settings";
      if (history?.rows === "fits" && Number.isInteger(rows.at(-1)?.trees)) selection = treeCount(rows.at(-1).trees);
      score.replaceChildren(element("strong", format(validated.validation_segment_macro_f1)),
        " validation interval macro-F1 ", element("span", "· " + selection));
    }
    const outcomes = result?.repeat_outcomes || [];
    const seeds = outcomes.length ? outcomes.map(row => row.seed) :
      [...new Set((records || []).map(row => row.seed).filter(Number.isInteger))];
    const selectedSeed = result?.selected_seed;
    for (const [index, value] of seeds.entries()) {
      const label = "Seed " + value + (value === selectedSeed ? " (used for this model)" : "");
      const item = seed.options[index];
      if (!item) seed.append(option(String(value), label));
      else if (item.value !== String(value) || item.text !== label) item.replaceWith(option(String(value), label));
    }
    while (seed.options.length > seeds.length) seed.lastElementChild.remove();
    // Follow the repeat being trained until the reader explicitly chooses one.
    if (seeds.length) {
      const automatic = seeds.includes(selectedSeed) ? selectedSeed : seeds.at(-1);
      seed.value = chosenSeed && seeds.includes(Number(chosenSeed)) ? chosenSeed : String(automatic);
    }
    seedLabel.hidden = seeds.length < 2;
    function draw() {
      const selected = (records || []).filter(row => !seeds.length || row.seed === Number(seed.value));
      const last = selected.at(-1);
      const outcome = outcomes.find(item => item.seed === Number(seed.value));
      const reason = outcome?.reason ?? last?.stop_reason;
      const summary = repeatEnding(reason, outcome?.step ?? last?.[history.field], policy);
      const best = outcome?.best_step ?? last?.best_step;
      const unit = reason === "update_limit" ? "update" : "epoch";
      const message = summary && best ? summary + ` Best validation score at ${unit} ${best}.` : summary;
      if (ending.textContent !== message) ending.textContent = message;
      ending.hidden = !message;
      if (!selected.length || (history.rows === "fits" && selected.length === 1)) {
        for (const view of figures.values()) view.inspector.hide();
        figures.clear();
        drawn = null;
        if ((policy?.budget === "fit" || fit) && validated && !result?.history_note) charts.replaceChildren();
        else charts.replaceChildren(element("p", result?.history_note || "The run has no recorded metric values to plot."));
        return;
      }
      if (drawn?.rows === rows && drawn.seed === seed.value && drawn.count === selected.length) return;
      const resetPage = drawn?.seed !== seed.value;
      drawn = {rows, seed: seed.value, count: selected.length};
      const series = chartSeries(history, selected, fromSavedWeights);
      const axis = horizontalAxis(history, selected);
      for (const [key, view] of figures) {
        if (!series.some(metric => metric.field === key)) {
          view.inspector.hide();
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
        const drawing = plot(metric, axis, gutter);
        if (!view) {
          const figure = element("figure"), values = valuesTable(history, metric), inspector = chartInspector();
          figure.append(drawing.caption, inspector.container, values.details);
          charts.insertBefore(figure, charts.children[index] || null);
          view = {figure, values, inspector};
          figures.set(metric.field, view);
        } else {
          view.figure.querySelector("figcaption").replaceWith(drawing.caption);
        }
        view.inspector.update(drawing, metric, axis, gutter, resetPage);
        view.values.update(metric.points, resetPage);
      }
    }
    seed.onchange = () => { chosenSeed = seed.value; draw(); };
    draw();
  }
  return {show, clear};
}
