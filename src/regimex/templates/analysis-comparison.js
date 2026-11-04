"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const data = JSON.parse($("comparison-data").textContent);
  const svg = $("chart");
  const width = 1000;
  const left = 82;
  const right = 976;
  const top = 24;
  const priceBottom = 192;
  const laneY = [232, 280];
  const differenceY = 328;
  const count = data.time_grid.step_count;
  const states = {
    growth: {label: "Growth", symbol: "↑", fill: "#d6eadc"},
    decline: {label: "Decline", symbol: "↓", fill: "#f3d9ce"},
    sideways: {label: "Sideways", symbol: "→", fill: "#e1e5ea"},
    no_trades: {label: "No trades", symbol: "Ø", fill: "url(#unavailable)"},
    insufficient_data: {label: "Insufficient data", symbol: "?", fill: "#f5f6f7"},
  };
  let selected = 0;
  let start = 0;
  let stop = count;
  let drag = null;
  let brush = null;
  let cursor = null;
  const number = (value) => value.toLocaleString(undefined, {maximumSignificantDigits: 6});
  const time = (step) => data.time_grid.origin + step * data.time_grid.bin_width;
  const x = (step) => left + (step - start) / (stop - start) * (right - left);
  const stateLabel = (state) => `${states[state].symbol} ${states[state].label}`;

  function element(tag, attributes = {}, text = null) {
    const result = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attributes).forEach(([name, value]) => result.setAttribute(name, String(value)));
    if (text !== null) result.textContent = text;
    return result;
  }
  function text(value, attributes) {
    svg.append(element("text", attributes, value));
  }
  function line(x1, y1, x2, y2, className) {
    svg.append(element("line", {x1, y1, x2, y2, class: className}));
  }
  function visible(item) {
    return item.start < stop && item.stop > start;
  }
  function band(item, y, height, fill, className = "") {
    const a = Math.max(start, item.start);
    const b = Math.min(stop, item.stop);
    const rect = element("rect", {
      x: x(a), y, width: x(b) - x(a), height, fill, class: className,
    });
    svg.append(rect);
    return rect;
  }
  function definePatterns() {
    const pattern = element("pattern", {id: "unavailable", width: 8, height: 8, patternUnits: "userSpaceOnUse"});
    pattern.append(
      element("rect", {width: 8, height: 8, fill: "#f0f2f5"}),
      element("path", {d: "M-2 2 L2 -2 M0 8 L8 0 M6 10 L10 6", stroke: "#b8c1cb", "stroke-width": 1}),
    );
    const defs = element("defs");
    defs.append(pattern);
    svg.append(defs);
  }
  function paintPrices(series) {
    const known = series.prices.slice(start, stop).filter((value) => value !== null);
    if (!known.length) {
      text("No observed prices in this range", {x: (left + right) / 2, y: 108, "text-anchor": "middle"});
      return;
    }
    // Normalize before subtraction so every accepted finite price remains plottable.
    const scale = Math.max(...known);
    let low = Math.min(...known) / scale;
    let high = Math.max(...known) / scale;
    if (low === high) {
      low -= 0.01;
      high = Math.min(high + 0.01, Number.MAX_VALUE / scale);
    }
    const y = (value) => priceBottom - (value / scale - low) / (high - low) * (priceBottom - top);
    for (let tick = 0; tick <= 4; tick++) {
      const value = low + (high - low) * tick / 4;
      const position = priceBottom - tick / 4 * (priceBottom - top);
      line(left, position, right, position, "grid");
      text(number(value * scale), {x: left - 8, y: position + 5, "text-anchor": "end"});
    }
    const paths = {observed: [], carried: []};
    let previous = null;
    for (let step = start; step < stop; step++) {
      const price = series.prices[step];
      if (price === null) {
        previous = null;
        continue;
      }
      const point = `${x(step)} ${y(price)}`;
      const state = series.price_states[step] === "carried" ? "carried" : "observed";
      if (previous !== null) paths[state].push(`M${previous} L${point}`);
      else svg.append(element("circle", {cx: x(step), cy: y(price), r: 2.5, fill: "#283747"}));
      previous = point;
    }
    for (const [state, segments] of Object.entries(paths)) {
      if (segments.length) svg.append(element("path", {d: segments.join(" "), class: `price ${state}`}));
    }
  }
  function paintIntervals(series) {
    series.intervals.forEach((intervals, index) => {
      text(index === 0 ? "A" : "B", {x: left - 18, y: laneY[index] + 23, "text-anchor": "end"});
      intervals.filter(visible).forEach((interval) => {
        const style = states[interval.state];
        const rect = band(interval, laneY[index], 34, style.fill);
        rect.setAttribute("stroke", "#fff");
        rect.append(element("title", {}, `${stateLabel(interval.state)} · [${interval.start}, ${interval.stop})`));
        const a = Math.max(start, interval.start);
        const b = Math.min(stop, interval.stop);
        if (x(b) - x(a) > 24)
          text(style.symbol, {x: (x(a) + x(b)) / 2, y: laneY[index] + 23, "text-anchor": "middle"});
      });
    });
  }
  function differences(series) {
    const items = series.differences.filter(visible).map((item) => ({
      ...item, start: Math.max(start, item.start), stop: Math.min(stop, item.stop),
    }));
    const totals = {direction: 0, availability: 0};
    $("differences").replaceChildren();
    items.forEach((item) => {
      totals[item.kind] += item.stop - item.start;
      const fill = item.kind === "direction" ? "#e9bc68" : "url(#unavailable)";
      const rect = band(item, differenceY, 18, fill);
      rect.append(element("title", {}, `${item.kind} difference · [${item.start}, ${item.stop})`));
      const row = document.createElement("tr");
      [`[${item.start}, ${item.stop})`, stateLabel(item.first), stateLabel(item.second), item.kind].forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      });
      $("differences").append(row);
    });
    $("differences-empty").hidden = items.length > 0;
    $("differences-table").hidden = items.length === 0;
    $("difference-summary").textContent = items.length
      ? `${totals.direction.toLocaleString()} steps with different directions · ${totals.availability.toLocaleString()} with different availability`
      : "Predictions agree in this range.";
  }
  function inspect(step) {
    const series = data.series[selected];
    step = Math.max(start, Math.min(stop - 1, step));
    $("inspect-step").value = String(step);
    const labels = series.intervals.map((intervals) =>
      stateLabel(intervals.find((item) => item.start <= step && item.stop > step).state));
    const price = series.prices[step];
    const priceText = price === null ? "unavailable" : `${number(price)} (${series.price_states[step]})`;
    $("inspection").textContent =
      `Step ${step} · Time ${number(time(step))} · Price ${priceText}` +
      ` · A: ${labels[0]} · B: ${labels[1]}`;
    if (cursor) {
      cursor.setAttribute("x1", x(step));
      cursor.setAttribute("x2", x(step));
    }
  }
  function render() {
    const series = data.series[selected];
    const instrument = series.instrument;
    svg.replaceChildren(
      element("title", {id: "svg-title"}, `${instrument.instrument_id}: prices and analysis intervals`),
      element("desc", {id: "svg-description"}, "The original price grid is shared by predictions A and B. The bottom row marks differences."),
    );
    definePatterns();
    $("units").textContent =
      `Price: ${instrument.quote_asset} / ${instrument.base_asset} · Time: ${data.time_grid.unit}` +
      ` · Step range [${start}, ${stop})`;
    paintPrices(series);
    paintIntervals(series);
    text("Differences", {x: left - 10, y: differenceY + 14, "text-anchor": "end", class: "small-label"});
    differences(series);
    line(left, priceBottom, right, priceBottom, "axis");
    for (let tick = 0; tick <= 4; tick++) {
      const step = start + (stop - start) * tick / 4;
      const position = x(step);
      line(position, 351, position, 357, "axis");
      text(number(time(step)), {x: position, y: 377, "text-anchor": tick === 4 ? "end" : tick === 0 ? "start" : "middle"});
    }
    text(`Time (${data.time_grid.unit})`, {x: (left + right) / 2, y: 399, "text-anchor": "middle"});
    cursor = element("line", {x1: left, x2: left, y1: top, y2: 347, class: "cursor"});
    svg.append(cursor);
    $("range-start").value = String(start);
    $("range-stop").value = String(stop);
    $("range-start").max = String(count - 1);
    $("range-stop").max = String(count);
    $("inspect-step").min = String(start);
    $("inspect-step").max = String(stop - 1);
    $("range-error").hidden = true;
    inspect(start);
  }
  function setRange(a, b) {
    if (!Number.isInteger(a) || !Number.isInteger(b) || a < 0 || b > count || a >= b) {
      $("range-error").textContent = `Choose integer steps with 0 ≤ start < stop ≤ ${count}.`;
      $("range-error").hidden = false;
      return;
    }
    start = a;
    stop = b;
    render();
  }
  function pointerStep(event) {
    const box = svg.getBoundingClientRect();
    const position = (event.clientX - box.left) / box.width * width;
    return Math.max(start, Math.min(stop, start + (position - left) / (right - left) * (stop - start)));
  }
  svg.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    drag = {step: pointerStep(event), pointer: event.pointerId};
    svg.setPointerCapture(event.pointerId);
    brush = element("rect", {x: x(drag.step), y: top, width: 0, height: 323, class: "brush"});
    svg.append(brush);
  });
  svg.addEventListener("pointermove", (event) => {
    const step = pointerStep(event);
    if (drag) {
      brush.setAttribute("x", x(Math.min(step, drag.step)));
      brush.setAttribute("width", Math.abs(x(step) - x(drag.step)));
    } else inspect(Math.floor(step));
  });
  svg.addEventListener("pointerup", (event) => {
    if (!drag || event.pointerId !== drag.pointer) return;
    const first = drag.step;
    const last = pointerStep(event);
    drag = null;
    brush.remove();
    brush = null;
    svg.releasePointerCapture(event.pointerId);
    if (Math.abs(first - last) >= 1) setRange(Math.floor(Math.min(first, last)), Math.ceil(Math.max(first, last)));
    else inspect(Math.floor(last));
  });
  svg.addEventListener("pointercancel", () => {
    drag = null;
    if (brush) brush.remove();
    brush = null;
  });
  $("instrument").addEventListener("change", () => {
    selected = Number($("instrument").value);
    render();
  });
  $("apply-range").addEventListener("click", () =>
    setRange(Number($("range-start").value), Number($("range-stop").value)));
  $("reset-range").addEventListener("click", () => setRange(0, count));
  $("inspect-step").addEventListener("input", () => inspect(Number($("inspect-step").value)));
  data.series.forEach((series, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = series.instrument.instrument_id;
    $("instrument").append(option);
  });
  const provenance = [
    ["Source", `${data.source.source_id} · ${data.source.source_version}`],
    ["Run", data.source.run_id],
    ["Snapshot", data.snapshot_id],
    ["Input SHA-256", data.snapshot_sha256],
    ["Renderer", data.renderer_version],
  ];
  if (JSON.stringify(data.analyses[0].analysis) !== JSON.stringify(data.analyses[1].analysis)) {
    $("policy-note").textContent = "These analyses use different interval settings; availability and minimum interval length can differ.";
    $("policy-note").hidden = false;
  }
  data.analyses.forEach((analysis, index) => {
    const label = index === 0 ? "A" : "B";
    provenance.push(
      [`${label} analysis`, analysis.analysis_id],
      [`${label} model`, `${analysis.model.model_version_id} · ${analysis.architecture_id}`],
      [`${label} model SHA-256`, analysis.model.bundle_sha256],
      [`${label} result SHA-256`, analysis.result_sha256],
      [`${label} decoder / profile`, `${analysis.decoder_version} · ${analysis.analysis_profile_id}`],
      [`${label} interval settings`, `${analysis.analysis.min_duration} minimum steps · ${analysis.analysis.minimum_observations} required prices · ${number(analysis.analysis.minimum_price_coverage * 100)}% price coverage`],
    );
  });
  provenance.forEach(([key, value]) => {
    const term = document.createElement("dt");
    term.textContent = key;
    const description = document.createElement("dd");
    description.textContent = value;
    $("provenance").append(term, description);
  });
  render();
})();
