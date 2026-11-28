"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const report = JSON.parse($("report-data").textContent);
  const samples = report.samples;
  const blind = report.kind === "reference";
  const predicted = report.kind === "prediction";
  const names = ["sideways", "growth", "decline", "unclassified"];
  let colors;
  function setColors() {
    colors = reportPlotColors();
    colors.unclassified = colors.no_trades;
    if (!reportIsDark()) Object.assign(colors, report.colors);
    for (const name of names) document.documentElement.style.setProperty("--direction-" + name, colors[name]);
  }
  setColors();
  const arrows = { sideways: "→", growth: "↗", decline: "↘", unclassified: "—" };
  const human = (value) => String(value).replaceAll("_", " ");
  const priceChange = (stats) =>
    stats.price_change_percent == null && stats.first_price_step !== stats.last_price_step
      ? "out of range"
      : fmt(stats.price_change_percent);
  const fmt = (value, digits = 2) =>
    value == null || !Number.isFinite(Number(value))
      ? "—"
      : Number(value).toLocaleString("en", { maximumFractionDigits: digits });
  const pct = (value) => (value == null ? "—" : fmt(100 * value));
  const countLabel = (count, noun) => fmt(count, 0) + " " + noun + (count === 1 ? "" : "s");
  const label = (interval) =>
    interval.direction == null
      ? human(interval.status)
      : arrows[interval.direction] + " " + interval.direction;
  const experiments = new Map();
  samples.forEach((sample, index) => {
    if (!experiments.has(sample.experiment_id)) experiments.set(sample.experiment_id, []);
    experiments.get(sample.experiment_id).push(index);
  });
  const experimentOrder = predicted
    ? report.summary.experiments.map((experiment) => experiment.experiment_id)
    : [...experiments.keys()];
  const experimentNames = new Map(
    experimentOrder.map((id, index) => [id, "Experiment " + (index + 1)]),
  );
  const sources = new Map((predicted ? report.summary.sources : []).map(item => [item.experiment_id, item]));
  experimentOrder.forEach((id) => {
    const indices = experiments.get(id);
    if (!indices) return;
    const option = document.createElement("option");
    option.value = id;
    const source = sources.get(id);
    const label = source && id !== source.snapshot_id
      ? source.source_namespace + " · " + source.snapshot_id + " · " + id.slice(0, 8)
      : experimentNames.get(id);
    option.textContent = label + " · " + countLabel(indices.length, "instrument");
    $("experiment").append(option);
  });
  function download(name, path) {
    const link = document.createElement("a");
    link.className = "download";
    link.href = path;
    link.textContent = name;
    $("downloads").append(link);
  }
  function cell(row, value) {
    const item = document.createElement("td");
    item.textContent = value;
    row.append(item);
    return item;
  }
  if (predicted) {
    $("pause-control").hidden = false;
    const summary = report.summary;
    $("eyebrow").textContent = "Regimex / Snapshot analysis";
    $("purpose").textContent = "Model predictions";
    download("Summary JSON", "summary.json");
    download("Raw snapshots", "snapshots.json");
    download("Model results", "results.json");
    download("Example data", "examples.json");
    $("dataset-counts").textContent =
      countLabel(summary.experiment_count, "experiment") + " · " +
      countLabel(summary.totals.instrument_sequences, "instrument");
  } else {
    $("dataset-counts").textContent =
      countLabel(experiments.size, "experiment") + " · " + countLabel(samples.length, "instrument");
    download("Observations", "observations.json");
    $("eyebrow").textContent = "Regimex / Independent review";
    $("title").textContent = "Review the price history";
    $("purpose").textContent = "Observations for human annotation";
    $("reference-instructions").hidden = false;
    download("Annotation draft", "annotation-draft.json");
  }
  function selectInstruments(instrumentId) {
    $("sample").replaceChildren();
    const experimentId = $("experiment").value;
    const indices = experiments.get(experimentId);
    indices.forEach((index) => {
      const sample = samples[index];
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = sample.instrument_id;
      if (sample.instrument_id === instrumentId) option.selected = true;
      $("sample").append(option);
    });
  }
  if (!blind)
    names.forEach((name) => {
      const item = document.createElement("span");
      const dot = document.createElement("span");
      dot.className = "dot";
      dot.style.background = "var(--direction-" + name + ")";
      item.append(dot, document.createTextNode(arrows[name] + " " + name));
      item.title = {
        sideways: "No sustained direction at this scale",
        growth: "Sustained upward movement",
        decline: "Sustained downward movement",
        unclassified: "Trading pause or insufficient observations",
      }[name];
      $("legend").append(item);
    });
  if (typeof Plotly === "undefined") {
    $("explorer").textContent = "Keep plotly.min.js next to index.html to display the plots.";
    return;
  }
  const guides = [
    "<b>Price</b> · solid: last transaction · dotted: bid · dashed: ask",
    "<b>Log return</b> · 100 × ln(current price / previous price)",
    "<b>Activity</b> · solid: trades · dashed: buys · dotted: sells",
    "<b>Volume</b> · same line styles as activity",
    "<b>Unavailable values</b> · × known absence · | unknown observation",
  ];
  const compactGuides = [
    "<b>Price</b> · last, bid, ask",
    "<b>Log return</b> (%)",
    "<b>Activity</b> · trades, buys, sells",
    "<b>Volume</b> · trades, buys, sells",
    "<b>Unavailable</b> · × absence · | unknown",
  ];
  let panelCount = 4;
  let current;
  let renderGeneration = 0;
  function zoomRange(interval) {
    const update = {};
    for (let i = 0; i < panelCount; i++)
      update["xaxis" + (i ? i + 1 : "") + ".range"] = [interval.start - 0.5, interval.stop - 0.5];
    Plotly.relayout("explorer", update);
  }
  function zoom(interval) {
    zoomRange(interval);
    const stats = interval.statistics;
    const price = priceChange(stats) + (stats.price_change_percent == null ? "" : "%");
    const activity = stats.trade_active_fraction == null
      ? "Trade counts unavailable"
      : pct(stats.trade_active_fraction) + "% of steps with known counts had trades";
    $("focus-summary").hidden = false;
    $("focus-summary").textContent = [
      "[" + interval.start + ", " + interval.stop + ") " + label(interval),
      countLabel(interval.stop - interval.start, "step"),
      "change " + price,
      activity,
    ].join(" · ");
  }
  function updatePauseList() {
    const enabled = $("trend-pauses").checked;
    renderTrendPauseList(enabled ? current.trend_pauses : [], enabled, current.instrument_id, (pause) => {
      zoomRange(pause);
      $("focus-summary").hidden = true;
    });
  }
  function renderDetails(steps, intervals, paired) {
    const classified = intervals.reduce(
      (sum, interval) => sum + (interval.direction == null ? 0 : interval.stop - interval.start),
      0,
    );
    $("sample-meta").textContent =
      countLabel(steps.length, "step") + " · price: " + current.quote_asset + " / " +
      current.base_asset + " · volume: " + current.base_asset + " per step";
    const grid = current.time_grid;
    $("time-guide").textContent =
      "Step width: " + grid.bin_width + " " + grid.unit +
      " · history [" + grid.origin + ", " + grid.completed_horizon + ")";
    const provenance = [["Input SHA-256", blind ? report.experiment_hash : sources.get(current.experiment_id).export_sha256]];
    if (!blind) {
      const source = current.source;
      provenance.push(
        ["Source", source.source_id + " · " + source.source_version],
        ["Snapshot", source.snapshot_id],
        ["Run", source.run_id === source.snapshot_id ? null : source.run_id],
        ["Model", report.summary.model.model_version_id],
        ["Model bundle SHA-256", report.summary.model.bundle_sha256],
        ["Analysis", current.prediction.analysis_id],
        ["Result SHA-256", current.prediction.result_sha256],
        ["Examples", report.summary.sample_experiment_ids.length === report.summary.experiment_count
          ? "All experiments" : report.summary.selection],
        ["Dataset SHA-256", report.summary.dataset_sha256],
      );
    }
    $("provenance").replaceChildren();
    provenance.forEach(([name, value]) => {
      if (value == null) return;
      const term = document.createElement("dt");
      term.textContent = name;
      const description = document.createElement("dd");
      description.textContent = value;
      $("provenance").append(term, description);
    });
    $("class-summary").textContent = blind
      ? ""
      : countLabel(intervals.length, "interval") +
        " · " +
        pct(classified / steps.length) +
        "% of steps classified";
    const analysis = blind ? report.analysis_config : current.analysis_config;
    $("config").textContent = JSON.stringify(analysis, null, 2);
    $("timeline").replaceChildren();
    $("intervals").replaceChildren();
    $("timeline").hidden = blind || paired;
    $("interval-wrap").hidden = blind;
    $("focus-summary").textContent = "";
    $("focus-summary").hidden = true;
    intervals.forEach((interval) => {
      const description = "[" + interval.start + ", " + interval.stop + ") " + label(interval);
      if (!paired) {
        const button = document.createElement("button");
        const fraction = (interval.stop - interval.start) / steps.length;
        button.style.width = 100 * fraction + "%";
        button.style.borderTopColor = "var(--direction-" + (interval.direction || "unclassified") + ")";
        button.textContent = fraction >= 0.09 ? label(interval) : "";
        button.title = description;
        button.setAttribute("aria-label", description);
        button.onclick = () => zoom(interval);
        $("timeline").append(button);
      }
      const row = document.createElement("tr");
      const link = document.createElement("button");
      link.textContent = "[" + interval.start + ", " + interval.stop + ")";
      link.setAttribute("aria-label", "Zoom to " + description);
      link.onclick = () => zoom(interval);
      cell(row, "").append(link);
      const stats = interval.statistics;
      [
        label(interval),
        priceChange(stats),
        pct(stats.return_std),
        fmt(stats.trade_count, 0),
        pct(stats.trade_active_fraction),
        pct(stats.price_coverage),
        pct(stats.trade_count_coverage),
      ].forEach((value) => cell(row, value));
      $("intervals").append(row);
    });
  }
  async function render(generation, visibleRange, preserveDetails) {
    current = samples[Number($("sample").value)];
    const observations = current.observations;
    const steps = current.steps;
    const intervals = current.intervals;
    const showPauses = predicted && $("trend-pauses").checked;
    const pauses = showPauses ? current.trend_pauses : [];
    const overlay = trendPauseOverlay(pauses, steps.length, (step) => step - 0.5, colors.sideways);
    const group = experiments.get(current.experiment_id);
    const paired = predicted && group.length > 1;
    const compact = $("explorer").clientWidth < 600;
    const missingFields = Object.keys(observations).filter(
      (field) => observations[field].some((value) => value === null),
    );
    panelCount = missingFields.length ? 5 : 4;
    const panelDomains = missingFields.length
      ? [
          [0.8, 0.99],
          [0.61, 0.73],
          [0.41, 0.54],
          [0.21, 0.34],
          [0, 0.14],
        ]
      : [
          [0.77, 0.99],
          [0.52, 0.69],
          [0.27, 0.44],
          [0.02, 0.19],
        ];
    const domains = paired ? panelDomains.map((domain) => domain.map((value) => value * 0.86)) : panelDomains;
    if (!preserveDetails) {
      renderDetails(steps, intervals, paired);
      if (predicted) updatePauseList();
    }
    const traces = [];
    function line(name, values, panel, color, extras = {}) {
      const suffix = panel ? String(panel + 1) : "";
      // A single value between gaps has no line segment to draw.
      const markerSizes = values.map((value, i) =>
        value != null && values[i - 1] == null && values[i + 1] == null ? 5 : 0);
      traces.push({
        type: "scatter",
        mode: markerSizes.some((size) => size > 0) ? "lines+markers" : "lines",
        marker: {color, size: markerSizes},
        name,
        x: steps,
        y: values,
        xaxis: "x" + suffix,
        yaxis: "y" + suffix,
        connectgaps: false,
        showlegend: false,
        line: { color, width: 1.2 },
        hovertemplate: "Step %{x}<br>%{y:.6g}<extra>" + name + "</extra>",
        ...extras,
      });
    }
    line("Last transaction price", observations.last_price, 0, colors.price, overlay.notes ? {
      text: overlay.notes,
      hovertemplate: "Step %{x}<br>%{y:.6g}%{text}<extra>Last transaction price</extra>",
    } : {});
    line("Best bid", observations.best_bid, 0, colors.growth, {
      line: { color: colors.growth, dash: "dot", width: 0.8 },
      visible: $("quotes").checked,
    });
    line("Best ask", observations.best_ask, 0, colors.decline, {
      line: { color: colors.decline, dash: "dash", width: 0.8 },
      visible: $("quotes").checked,
    });
    const returns = observations.last_price.map((price, index, prices) => {
      const previous = prices[index - 1];
      if (price == null || previous == null) return null;
      const change = (price - previous) / previous;
      // Separate logs keep extreme ratios finite, while log1p preserves small ticks.
      const logReturn = Math.abs(change) <= 0.5
        ? Math.log1p(change)
        : Math.log(price) - Math.log(previous);
      return 100 * logReturn;
    });
    line("Adjacent log return (%)", returns, 1, colors.ink);
    ["trade", "buy_order", "sell_order"].forEach((field, index) => {
      const style = {
        line: {
          color: [colors.ink, colors.growth, colors.decline][index],
          dash: ["solid", "dash", "dot"][index],
          width: 1.2,
        },
      };
      line(human(field) + " count", observations[field + "_count"], 2, style.line.color, style);
      line(human(field) + " volume", observations[field + "_volume"], 3, style.line.color, style);
    });
    missingFields.forEach((field, index) => {
      const states =
        field === "best_bid"
          ? current.availability.bid_state
          : field === "best_ask"
            ? current.availability.ask_state
            : field === "last_price"
              ? current.availability.price_state
              : null;
      const absent = steps.filter((_, step) => observations[field][step] === null);
      const absentStates = absent.map((step) => states?.[step]);
      const knownAbsence = absentStates.map((state) => state === "empty" || state === "no_execution");
      traces.push({
        type: "scatter",
        mode: "markers",
        name: human(field) + " unavailable",
        x: absent,
        y: absent.map(() => index),
        customdata: absentStates.map((state) =>
          state === "empty"
            ? "Observed empty book side"
            : state === "no_execution"
              ? "No execution in this bin"
              : "Unknown observation",
        ),
        xaxis: "x5",
        yaxis: "y5",
        showlegend: false,
        marker: {
          color: knownAbsence.map((known) => (known ? colors.no_trades : colors.unknown)),
          size: 6,
          symbol: knownAbsence.map((known) => (known ? "x-open" : "line-ns-open")),
        },
        hovertemplate: "Step %{x}<br>" + human(field) + "<br>%{customdata}<extra></extra>",
      });
    });
    const layout = {
      height: (missingFields.length ? 1150 : 940) + (paired ? 120 : 0),
      uirevision: preserveDetails ? $("explorer").layout?.uirevision : generation,
      margin: { l: compact ? 85 : 140, r: compact ? 10 : 25, t: 30, b: 60 },
      paper_bgcolor: colors.paper, plot_bgcolor: colors.plot,
      modebar: {color: colors.no_trades, activecolor: colors.ink, bgcolor: colors.paper},
      hoverlabel: {bgcolor: colors.paper, font: {color: colors.ink}},
      hovermode: paired ? "closest" : "x unified",
      font: { family: "system-ui, sans-serif", size: 12, color: colors.ink },
      annotations: (compact ? compactGuides : guides).slice(0, panelCount).map((text, index) => ({
        text,
        xref: "paper",
        yref: "paper",
        x: 0,
        y: domains[index][1],
        xanchor: "left",
        yanchor: "bottom",
        yshift: 10,
        showarrow: false,
        font: { size: 11 },
      })),
      shapes: intervals.map((interval) => ({
        type: "rect",
        xref: "x",
        yref: "paper",
        x0: interval.start - 0.5,
        x1: interval.stop - 0.5,
        y0: domains[0][0],
        y1: domains[0][1],
        fillcolor: colors[interval.direction || "unclassified"],
        opacity: 0.1,
        line: { width: 0 },
        layer: "below",
      })),
    };
    layout.shapes.push(...overlay.shapes);
    [
      "Price (quote / base)",
      "Log return (%)",
      "Count per step",
      "Base units per step",
      "Missing fields",
    ].slice(0, panelCount).forEach((title, index) => {
      const suffix = index ? String(index + 1) : "";
      layout["xaxis" + suffix] = {
        domain: [0, 1],
        anchor: "y" + suffix,
        matches: index ? "x" : undefined,
        range: visibleRange ? [...visibleRange] : [-0.5, steps.length - 0.5],
        gridcolor: colors.grid,
        title: index === panelCount - 1 ? { text: "Step index" } : undefined,
      };
      layout["yaxis" + suffix] = {
        domain: domains[index],
        anchor: "x" + suffix,
        title: { text: title },
        gridcolor: colors.grid,
        rangemode: index === 2 || index === 3 ? "tozero" : "normal",
      };
    });
    if (missingFields.length)
      Object.assign(layout.yaxis5, {
        tickmode: "array",
        tickvals: missingFields.map((_, index) => index),
        ticktext: missingFields.map(human),
        range: [missingFields.length - 0.5, -0.5],
        title: undefined,
        automargin: compact,
      });
    if (paired) {
      const lanes = instrumentLanes(
        group.map((index) => samples[index]), current.instrument_id,
        colors,
        (step) => step - 0.5,
      );
      const suffix = panelCount + 1;
      layout.barmode = "overlay";
      layout["yaxis" + suffix] = { ...lanes.axis, domain: [0.90, 1], anchor: "x" };
      for (const trace of lanes.traces) {
        traces.push({ ...trace, xaxis: "x", yaxis: "y" + suffix, showlegend: false });
      }
    }
    domains.slice(0, 4).forEach((domain, index) => {
      const axis = "y" + (index ? index + 1 : "");
      if (
        !traces.some(
          (trace) =>
            trace.yaxis === axis &&
            trace.visible !== false &&
            trace.y.some((value) => value !== null),
        )
      )
        layout.annotations.push({
          text: "No observed values",
          xref: "paper",
          yref: "paper",
          x: 0.5,
          y: (domain[0] + domain[1]) / 2,
          showarrow: false,
        });
    });
    history.replaceState(
      null,
      "",
      "#" +
        new URLSearchParams({
          experiment: current.experiment_id,
          instrument: current.instrument_id,
        }),
    );
    $("explorer").removeAllListeners?.("plotly_click");
    $("explorer").style.height = layout.height + "px";
    await Plotly.react("explorer", traces, layout, {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["toImage"],
    });
    if (paired && generation === renderGeneration) {
      $("explorer").on("plotly_click", (event) => {
        const index = group[instrumentLanePoint(event)?.y];
        if (index !== undefined && $("sample").value !== String(index)) {
          $("sample").value = String(index);
          update(true);
        }
      });
    }
  }
  function update(keepZoom = false, preserveDetails = false) {
    const visibleRange = keepZoom ? $("explorer").layout?.xaxis.range : null;
    const generation = ++renderGeneration;
    $("render-error").hidden = true;
    $("explorer").hidden = false;
    render(generation, visibleRange, preserveDetails).catch((error) => {
      // A late failure from a previous selection cannot hide the current view.
      if (generation !== renderGeneration) return;
      $("render-error").textContent = "Could not display this selection: " + error.message;
      $("render-error").hidden = false;
      $("explorer").hidden = true;
      $("timeline").hidden = true;
      $("interval-wrap").hidden = true;
    });
  }
  const initial = new URLSearchParams(location.hash.slice(1));
  if (experiments.has(initial.get("experiment"))) {
    $("experiment").value = initial.get("experiment");
  }
  selectInstruments(initial.get("instrument"));
  $("sample").onchange = () => update(true);
  $("experiment").onchange = () => {
    selectInstruments(samples[Number($("sample").value)].instrument_id);
    update();
  };
  $("reset").onclick = () => update();
  $("quotes").onchange = () => update(true);
  $("trend-pauses").onchange = () => {
    updatePauseList();
    update(true, true);
  };
  watchReportTheme(() => {
    setColors();
    update(true, true);
  });
  update();
})();
