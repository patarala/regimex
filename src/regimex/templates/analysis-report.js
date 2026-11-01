"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const data = JSON.parse($("analysis-data").textContent);
  const input = data.export;
  const result = data.result;
  const grid = input.time_grid;
  $("dataset-counts").textContent =
    "1 experiment · " + input.instruments.length + " instrument" +
    (input.instruments.length === 1 ? "" : "s");
  // Reconstruct named fields without changing nulls or observed zeros.
  input.observations = data.observations.flatMap((block) =>
    block.rows.map((values) => {
      const row = { instrument_id: block.instrument_id };
      data.observation_fields.forEach((field, index) => { row[field] = values[index]; });
      return row;
    }),
  );
  result.sequences = data.sequences.map((sequence) => ({
    ...sequence,
    intervals: sequence.intervals.map((item) => ({ ...item, instrument_id: sequence.instrument_id })),
  }));
  delete data.observation_fields;
  delete data.observations;
  delete data.sequences;

  const selector = $("instrument");
  const quotes = $("quotes");
  const colors = { growth: "#147d64", decline: "#b84c34", sideways: "#416ea7", no_trades: "#8a8e98", insufficient_data: "#a29b84" };
  const symbols = { growth: "↗", decline: "↘", sideways: "→", no_trades: "—", insufficient_data: "—" };
  const nodes = ["price", "timeline", "activity", "volume", "coverage", "probability"].map($);
  const escape = (value) => String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;");
  const coordinate = (step) => Math.min(grid.origin + step * grid.bin_width, grid.completed_horizon);
  const extent = [grid.origin, grid.completed_horizon];
  const sampleTime = (row) => (coordinate(row.step) + coordinate(row.step + 1)) / 2;
  let activeNodes = [];
  let synchronizing = false;
  let generation = 0;
  for (const instrument of input.instruments) {
    const option = document.createElement("option");
    option.value = instrument.instrument_id;
    option.textContent = instrument.instrument_id;
    selector.append(option);
  }

  function layout(title, unit, height) {
    const plotHeight = height - 75;
    return {
      title: { text: title.replace(/(.{26,40})\s/g, "$1<br>"), font: { size: 14 } },
      height: height + 110,
      margin: { l: 65, r: 20, t: 45, b: 140 },
      paper_bgcolor: "white", plot_bgcolor: "#fbfcfe",
      font: { family: "system-ui, sans-serif", color: "#17283d" },
      xaxis: {
        title: { text: escape(grid.unit.length > 24 ? grid.unit.slice(0, 24) + "…" : grid.unit) },
        range: extent, showgrid: true, gridcolor: "#e7edf4",
      },
      yaxis: {
        title: { text: unit.length > 36 ? unit.slice(0, 36) + "…" : unit },
        fixedrange: false, gridcolor: "#e7edf4",
      },
      // Leave room below the axis title for legends that wrap on narrow screens.
      legend: { orientation: "h", y: -70 / plotHeight, yanchor: "top" },
      hovermode: "x unified", uirevision: selector.value,
    };
  }
  const config = {
    responsive: true, displaylogo: false, scrollZoom: false,
    modeBarButtonsToRemove: ["toImage", "sendDataToCloud", "select2d", "lasso2d", "autoScale2d"],
  };
  function trace(rows, field, name, color, dash = "solid") {
    const state = { last_price: "price_state", best_bid: "bid_state", best_ask: "ask_state" }[field];
    return {
      type: "scatter", mode: "lines+markers", marker: { size: 2 }, name,
      x: rows.map(sampleTime), y: rows.map((row) => row[field]), connectgaps: false,
      line: { color, width: 1.7, dash },
      customdata: state ? rows.map((row) => row[state]) : undefined,
      hovertemplate: "%{x}<br>%{y}" + (state ? "<br>%{customdata}" : "") + "<extra>" + name + "</extra>",
    };
  }
  function unavailable(rows, fields) {
    return fields.map((field) => {
      const absent = rows.filter((row) => row[field] === null);
      const states = absent.map((row) => {
        const state = field === "best_bid" ? row.bid_state
          : field === "best_ask" ? row.ask_state : field === "last_price" ? row.price_state : null;
        return state === "empty" ? "Empty book side"
          : state === "no_execution" ? "No execution in this bin" : "Unknown observation";
      });
      return {
        type: "scatter", mode: "markers", name: field.replaceAll("_", " "),
        x: absent.map(sampleTime), y: absent.map(() => field.replaceAll("_", " ")),
        customdata: states, showlegend: false,
        marker: {
          size: 7,
          color: states.map((state) => state === "Unknown observation" ? "#7c3aed" : "#64748b"),
          symbol: states.map((state) => state === "Unknown observation" ? "line-ns-open" : "x-open"),
        },
        hovertemplate: "%{y}<br>%{x}<br>%{customdata}<extra></extra>",
      };
    });
  }

  async function draw(current) {
    const instrument = input.instruments.find((item) => item.instrument_id === selector.value);
    const sequence = result.sequences.find((item) => item.instrument_id === selector.value);
    const rows = input.observations.filter((row) => row.instrument_id === selector.value);
    const priceUnit = escape(instrument.quote_asset) + " / " + escape(instrument.base_asset);
    const volumeUnit = escape(instrument.base_asset);
    $("units").textContent =
      "Time: " + grid.unit + " · bin width: " + grid.bin_width + " · price: " +
      instrument.quote_asset + " / " + instrument.base_asset + " · volume: " +
      instrument.base_asset + " · history [" + grid.origin + ", " + grid.completed_horizon + ")";
    const price = [trace(rows, "last_price", "Last price", "#263b55")];
    if (quotes.checked) price.push(
      trace(rows, "best_bid", "Best bid", "#247961", "dot"),
      trace(rows, "best_ask", "Best ask", "#a14f48", "dash"),
    );
    const intervals = Object.keys(colors).map((key) => {
      const group = sequence.intervals.filter((interval) => (interval.direction || interval.status) === key);
      return {
        type: "bar", orientation: "h", name: symbols[key] + " " + key.replaceAll("_", " "),
        y: group.map(() => "Prediction"),
        x: group.map((interval) => coordinate(interval.stop) - coordinate(interval.start)),
        base: group.map((interval) => coordinate(interval.start)), width: .65,
        marker: { color: colors[key], line: { color: "white", width: 1 } },
        text: group.map(() => symbols[key]),
        customdata: group.map((interval) => "[" + coordinate(interval.start) + ", " +
          coordinate(interval.stop) + ")<br>bins [" + interval.start + ", " + interval.stop + ")"),
        textposition: "inside", insidetextanchor: "middle",
        hovertemplate: escape(symbols[key] + " " + key) + "<br>%{customdata}<extra></extra>",
        showlegend: group.length > 0,
      };
    });
    const timeline = layout("Predicted directions", "", 160);
    timeline.barmode = "overlay";
    timeline.yaxis = { fixedrange: true };
    const count = [
      trace(rows, "trade_count", "Executions", "#263b55"),
      trace(rows, "buy_order_count", "Accepted buys", "#247961"),
      trace(rows, "sell_order_count", "Accepted sells", "#a14f48"),
    ];
    const volume = [
      trace(rows, "trade_volume", "Executed volume", "#263b55"),
      trace(rows, "buy_order_volume", "Accepted buy volume", "#247961"),
      trace(rows, "sell_order_volume", "Accepted sell volume", "#a14f48"),
    ];
    const fields = ["last_price", "trade_count", "trade_volume", "buy_order_count", "sell_order_count", "buy_order_volume", "sell_order_volume", "best_bid", "best_ask"]
      .filter((field) => rows.some((row) => row[field] === null));
    const availability = layout("Unavailable: × known absence · | unknown", "", 230);
    availability.margin.l = 135;
    availability.yaxis = {
      type: "category", categoryorder: "array",
      categoryarray: fields.map((field) => field.replaceAll("_", " ")), autorange: "reversed",
    };
    const probabilities = result.class_names.map((name, index) => ({
      type: "scatter", mode: "lines", name: symbols[name] + " " + name,
      x: rows.map(sampleTime), y: sequence.probabilities.map((row) => row[index]),
      line: { color: colors[name], width: 1.6 },
      hovertemplate: "%{x}<br>%{y:.4f}<extra>" + name + "</extra>",
    }));
    const probability = layout("Model probabilities", "Probability", 290);
    probability.yaxis.range = [0, 1];
    const definitions = [
      [$("price"), price, layout("Execution price and historical quotes", priceUnit, 290)],
      [$("timeline"), intervals, timeline],
      [$("activity"), count, layout("Executions and accepted orders", "Count per bin", 290)],
      [$("volume"), volume, layout("Execution and accepted order volumes", volumeUnit + " per bin", 290)],
      ...(fields.length ? [[$("coverage"), unavailable(rows, fields), availability]] : []),
      [$("probability"), probabilities, probability],
    ];
    const drawn = definitions.map(([node]) => node);
    for (const node of nodes) {
      node.hidden = !drawn.includes(node);
      node.removeAllListeners?.("plotly_relayout");
    }
    await Promise.all(definitions.map(([node, traces, settings]) => {
      node.style.height = settings.height + "px";
      return Plotly.react(node, traces, settings, config);
    }));
    if (current !== generation) return;
    activeNodes = drawn;
    for (const node of activeNodes) {
      node.on("plotly_relayout", (change) => {
        if (synchronizing) return;
        let range;
        if (change["xaxis.autorange"] === true) range = extent;
        else if (change["xaxis.range"]) range = change["xaxis.range"];
        else if (change["xaxis.range[0]"] !== undefined) range = [change["xaxis.range[0]"], change["xaxis.range[1]"]];
        if (!range) return;
        synchronizing = true;
        Promise.all(activeNodes.filter((other) => other !== node)
          .map((other) => Plotly.relayout(other, { "xaxis.range": range })))
          .catch(failed).finally(() => { synchronizing = false; });
      });
    }
  }
  function failed(error) {
    $("plot-error").hidden = false;
    $("plot-error").textContent = "Could not display the plots: " + String(error);
  }
  function refresh() {
    const current = ++generation;
    $("plot-error").hidden = true;
    draw(current).catch((error) => { if (current === generation) failed(error); });
  }
  selector.addEventListener("change", refresh);
  quotes.addEventListener("change", refresh);
  $("reset").addEventListener("click", () => {
    synchronizing = true;
    Promise.all(activeNodes.map((node) => Plotly.relayout(node, { "xaxis.range": extent })))
      .catch(failed).finally(() => { synchronizing = false; });
  });
  // JSON.stringify normalizes -0 to 0. Preserve its sign for the content digest.
  function savedJSON(value) {
    if (typeof value === "number") return Object.is(value, -0) ? "-0.0" : JSON.stringify(value);
    if (value === null || typeof value !== "object") return JSON.stringify(value);
    if (Array.isArray(value)) return "[" + value.map(savedJSON).join(",") + "]";
    return "{" + Object.keys(value).map((key) => JSON.stringify(key) + ":" + savedJSON(value[key])).join(",") + "}";
  }
  $("download").addEventListener("click", () => {
    const blob = new Blob([savedJSON(data)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "regimex-analysis-" + result.analysis_id + ".json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  refresh();
})();
