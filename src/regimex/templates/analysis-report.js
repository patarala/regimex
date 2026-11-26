"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const packedJSON = $("analysis-data").textContent;
  const packed = JSON.parse(packedJSON);
  const fields = packed.observation_fields;
  // The same expansion is used for plotting numbers and downloading their exact JSON literals.
  function unpack({observations, sequences, export: input, result, report}) {
    input.observations = observations.map(([instrumentIndex, ...values]) => {
      const row = {instrument_id: input.instruments[instrumentIndex].instrument_id};
      fields.forEach((field, index) => { row[field] = values[index]; });
      return row;
    });
    result.sequences = sequences.map((sequence) => ({
      ...sequence,
      intervals: sequence.intervals.map((item) => ({...item, instrument_id: sequence.instrument_id})),
    }));
    return {export: input, result, report};
  }
  const data = unpack(packed);
  const input = data.export;
  const result = data.result;
  const grid = input.time_grid;
  $("dataset-counts").textContent =
    "1 experiment · " + input.instruments.length + " instrument" +
    (input.instruments.length === 1 ? "" : "s");

  const selector = $("instrument");
  const quotes = $("quotes");
  const pauseToggle = $("trend-pauses");
  const nodes = ["price", "timeline", "activity", "volume", "coverage", "probability"].map($);
  const axis = reportTimeAxis(grid, packed.last_bin_fraction);
  const coordinate = axis.boundary;
  const timeUnit = axis.unit;
  const extent = [coordinate(0), coordinate(grid.step_count)];
  let visibleRange = [...extent];
  const sampleTime = (row) => axis.center(row.step);
  const sequences = input.instruments.map((instrument) =>
    result.sequences.find((sequence) => sequence.instrument_id === instrument.instrument_id));
  const paired = sequences.length > 1;
  if (paired) $("price").before($("timeline"));
  let activeNodes = [];
  let synchronizing = false;
  let generation = 0;
  for (const instrument of input.instruments) {
    const option = document.createElement("option");
    option.value = instrument.instrument_id;
    option.textContent = instrument.instrument_id;
    selector.append(option);
  }

  function layout(colors, title, unit, height) {
    const plotHeight = height - 75;
    return {
      title: { text: title.replace(/(.{26,40})\s/g, "$1<br>"), font: { size: 14 } },
      height: height + 110,
      margin: { l: paired ? 84 : 65, r: 20, t: 45, b: 140 },
      paper_bgcolor: colors.paper, plot_bgcolor: colors.plot,
      modebar: {color: colors.no_trades, activecolor: colors.ink, bgcolor: colors.paper},
      hoverlabel: {bgcolor: colors.paper, font: {color: colors.ink}},
      font: { family: "system-ui, sans-serif", color: colors.ink },
      xaxis: {
        title: { text: plotLabel(timeUnit.length > 24 ? timeUnit.slice(0, 24) + "…" : timeUnit) },
        // Plotly mutates axis ranges during interactive zoom.
        range: [...visibleRange], showgrid: true, gridcolor: colors.grid,
      },
      yaxis: {
        title: { text: unit.length > 36 ? unit.slice(0, 36) + "…" : unit },
        fixedrange: false, gridcolor: colors.grid,
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
  function trace(rows, field, name, color, dash = "solid", hoverNotes) {
    const state = { last_price: "price_state", best_bid: "bid_state", best_ask: "ask_state" }[field];
    return {
      type: "scatter", mode: "lines+markers", marker: { size: 2 }, name,
      x: rows.map(sampleTime), y: rows.map((row) => row[field]), connectgaps: false,
      line: { color, width: 1.7, dash },
      customdata: state ? rows.map((row) => row[state]) : undefined,
      text: hoverNotes,
      hovertemplate: "%{x}<br>%{y}" + (state ? "<br>%{customdata}" : "") +
        (hoverNotes ? "%{text}" : "") + "<extra>" + name + "</extra>",
    };
  }

  function priceChart(colors, instrument, rows) {
    const pauses = pauseToggle.checked ? data.report.trend_pauses[selector.selectedIndex] : [];
    const unit = plotLabel(instrument.quote_asset) + " / " + plotLabel(instrument.base_asset);
    const settings = layout(colors, "Execution price and historical quotes", unit, 290);
    const {shapes, notes} = trendPauseOverlay(pauses, rows.length, coordinate, colors.sideways);
    settings.shapes = shapes;
    renderTrendPauseList(pauses, pauseToggle.checked, instrument.instrument_id,
      (pause) => selectInterval(instrument.instrument_id, pause.start, pause.stop));
    const traces = [trace(rows, "last_price", "Last price", colors.price, "solid", notes)];
    if (quotes.checked) traces.push(
      trace(rows, "best_bid", "Best bid", colors.growth, "dot"),
      trace(rows, "best_ask", "Best ask", colors.decline, "dash"),
    );
    return [$("price"), traces, settings];
  }
  function unavailable(colors, rows, fields) {
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
          color: states.map((state) => state === "Unknown observation" ? colors.unknown : colors.no_trades),
          symbol: states.map((state) => state === "Unknown observation" ? "line-ns-open" : "x-open"),
        },
        hovertemplate: "%{y}<br>%{x}<br>%{customdata}<extra></extra>",
      };
    });
  }

  async function draw(current) {
    const colors = reportPlotColors();
    const instrument = input.instruments[selector.selectedIndex];
    const sequence = sequences[selector.selectedIndex];
    const rows = selectedRows();
    const volumeUnit = plotLabel(instrument.base_asset);
    $("units").textContent =
      "Time: " + grid.unit + " · bin width: " + grid.bin_width + " · price: " +
      instrument.quote_asset + " / " + instrument.base_asset + " · volume: " +
      instrument.base_asset + " · history [" + grid.origin + ", " + grid.completed_horizon + ")";
    const lanes = instrumentLanes(sequences, selector.value, colors, coordinate, timeUnit);
    const timeline = layout(colors, "Predicted directions", "", paired ? 192 : 160);
    timeline.barmode = "overlay";
    timeline.hovermode = "closest";
    timeline.yaxis = lanes.axis;
    const count = [
      trace(rows, "trade_count", "Executions", colors.price),
      trace(rows, "buy_order_count", "Accepted buys", colors.growth),
      trace(rows, "sell_order_count", "Accepted sells", colors.decline),
    ];
    const volume = [
      trace(rows, "trade_volume", "Executed volume", colors.price),
      trace(rows, "buy_order_volume", "Accepted buy volume", colors.growth),
      trace(rows, "sell_order_volume", "Accepted sell volume", colors.decline),
    ];
    const fields = ["last_price", "trade_count", "trade_volume", "buy_order_count", "sell_order_count", "buy_order_volume", "sell_order_volume", "best_bid", "best_ask"]
      .filter((field) => rows.some((row) => row[field] === null));
    const availability = layout(colors, "Unavailable: × known absence · | unknown", "", 230);
    availability.margin.l = 135;
    availability.yaxis = {
      ...availability.yaxis, type: "category", categoryorder: "array",
      categoryarray: fields.map((field) => field.replaceAll("_", " ")), autorange: "reversed",
    };
    const probabilities = result.class_names.map((name, index) => ({
      type: "scatter", mode: rows.length === 1 ? "markers" : "lines", name: directionSymbols[name] + " " + name,
      x: rows.map(sampleTime), y: sequence.probabilities.map((row) => row[index]),
      marker: {color: colors[name]},
      line: { color: colors[name], width: 1.6 },
      hovertemplate: "%{x}<br>%{y:.4f}<extra>" + name + "</extra>",
    }));
    const probability = layout(colors, "Model probabilities", "Probability", 290);
    probability.yaxis.range = [0, 1];
    const definitions = [
      priceChart(colors, instrument, rows),
      [$("timeline"), lanes.traces, timeline],
      [$("activity"), count, layout(colors, "Executions and accepted orders", "Count per bin", 290)],
      [$("volume"), volume, layout(colors, "Execution and accepted order volumes", volumeUnit + " per bin", 290)],
      ...(fields.length ? [[$("coverage"), unavailable(colors, rows, fields), availability]] : []),
      [$("probability"), probabilities, probability],
    ];
    const drawn = definitions.map(([node]) => node);
    $("timeline").removeAllListeners?.("plotly_click");
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
    $("timeline").on("plotly_click", (event) => {
      const point = instrumentLanePoint(event);
      if (point) selectInterval(sequences[point.y].instrument_id, ...point.customdata);
    });
    for (const node of activeNodes) {
      node.on("plotly_relayout", (change) => {
        if (synchronizing) return;
        let range;
        if (change["xaxis.autorange"] === true) range = extent;
        else if (change["xaxis.range"]) range = change["xaxis.range"];
        else if (change["xaxis.range[0]"] !== undefined) range = [change["xaxis.range[0]"], change["xaxis.range[1]"]];
        if (!range) return;
        visibleRange = [...range];
        synchronizing = true;
        const targets = change["xaxis.autorange"] ? activeNodes : activeNodes.filter((other) => other !== node);
        Promise.all(targets
          .map((other) => Plotly.relayout(other, { "xaxis.range": [...range] })))
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
  function selectedRows() {
    return input.observations.filter((row) => row.instrument_id === selector.value);
  }
  function refreshPrice() {
    const instrument = input.instruments[selector.selectedIndex];
    const [node, traces, settings] = priceChart(reportPlotColors(), instrument, selectedRows());
    Plotly.react(node, traces, settings, config).catch(failed);
  }
  watchReportTheme(refresh);
  selector.addEventListener("change", refresh);
  quotes.addEventListener("change", refreshPrice);
  pauseToggle.addEventListener("change", refreshPrice);
  function setRange(range) {
    visibleRange = [...range];
    synchronizing = true;
    Promise.all(activeNodes.map((node) => Plotly.relayout(node, { "xaxis.range": [...range] })))
      .catch(failed).finally(() => { synchronizing = false; });
  }
  function selectInterval(instrument, start, stop) {
    const range = [coordinate(start), coordinate(stop)];
    if (selector.value === instrument) {
      setRange(range);
    } else {
      selector.value = instrument;
      visibleRange = range;
      refresh();
    }
  }
  for (const row of $("intervals").rows) {
    const instrument = row.cells[0].textContent;
    const bounds = row.cells[1];
    const button = document.createElement("button");
    button.type = "button";
    button.className = "interval-link";
    button.textContent = bounds.textContent;
    button.setAttribute("aria-label", `Zoom to ${instrument} steps ${bounds.textContent}`);
    button.addEventListener("click", () => {
      selectInterval(instrument, Number(row.dataset.start), Number(row.dataset.stop));
      selector.focus({preventScroll: true});
      $("observations").scrollIntoView({block: "start"});
    });
    bounds.replaceChildren(button);
  }
  $("reset").addEventListener("click", () => setRange(extent));
  $("download").addEventListener("click", () => {
    // JSON.parse erases the distinction between 1 and 1.0. Preserve the original
    // scalar tokens while expanding the compact rows for the digest-bound download.
    const tokens = /"(?:\\.|[^"\\])*"(\s*:)?|-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi;
    const literals = JSON.parse(packedJSON.replace(tokens, (token, key) => key ? token : JSON.stringify(token)));
    const json = JSON.stringify(unpack(literals)).replace(tokens, (token, key) => key ? token : JSON.parse(token));
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "regimex-analysis-" + result.analysis_id + ".json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  refresh();
})();
