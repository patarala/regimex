"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const report = JSON.parse($("report-data").textContent);
  const samples = report.samples;
  const blind = report.kind === "reference";
  const predicted = report.kind === "prediction";
  const names = ["sideways", "growth", "decline", "unclassified"];
  const arrows = { sideways: "→", growth: "↗", decline: "↘", unclassified: "—" };
  const human = (value) => String(value).replaceAll("_", " ");
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
  experimentOrder.forEach((id) => {
    const indices = experiments.get(id);
    if (!indices) return;
    const option = document.createElement("option");
    option.value = id;
    option.textContent = experimentNames.get(id) + " · " + countLabel(indices.length, "instrument");
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
    download("Input export", "export.json");
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
      dot.style.background = report.colors[name];
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
    "<b>Price</b> · black: last transaction · dotted blue: bid · dashed pink: ask",
    "<b>Log return</b> · 100 × ln(current price / previous price)",
    "<b>Activity</b> · solid gray: trades · dashed green: BUY · dotted red: SELL",
    "<b>Volume</b> · same line styles as activity",
    "<b>Unavailable values</b> · × known absence · | unknown observation",
  ];
  let panelCount = 4;
  let current;
  let renderGeneration = 0;
  function zoom(interval) {
    const update = {};
    for (let i = 0; i < panelCount; i++)
      update["xaxis" + (i ? i + 1 : "") + ".range"] = [interval.start - 0.5, interval.stop - 0.5];
    Plotly.relayout("explorer", update);
    const stats = interval.statistics;
    $("focus-summary").hidden = false;
    $("focus-summary").textContent =
      "[" +
      interval.start +
      ", " +
      interval.stop +
      ") " +
      label(interval) +
      " · " +
      (interval.stop - interval.start) +
      " steps · change " +
      fmt(stats.price_change_percent) +
      "% · active steps " +
      pct(stats.trade_active_fraction) +
      "%";
  }
  async function render() {
    current = samples[Number($("sample").value)];
    const observations = current.observations;
    const steps = current.steps;
    const intervals = current.intervals;
    const classified = intervals.reduce(
      (sum, interval) => sum + (interval.direction == null ? 0 : interval.stop - interval.start),
      0,
    );
    $("sample-meta").textContent =
      countLabel(steps.length, "step") + " · price: " + current.quote_asset + " / " +
      current.base_asset + " · volume: " + current.base_asset + " per step";
    const missingFields = Object.keys(observations).filter(
      (field) => observations[field].some((value) => value === null),
    );
    panelCount = missingFields.length ? 5 : 4;
    const domains = missingFields.length
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
    const grid = current.time_grid;
    $("time-guide").textContent =
      "Step width: " + fmt(grid.bin_width, 6) + " " + grid.unit +
      " · history [" + fmt(grid.origin, 6) + ", " + fmt(grid.completed_horizon, 6) + ")";
    const source = current.source;
    const sourceHash = blind ? report.experiment_hash : report.summary.sources.find(
      (item) => item.snapshot_id === current.experiment_id,
    ).export_sha256;
    const provenance = [
      ["Source", source.source_id + " · " + source.source_version],
      ["Snapshot", source.snapshot_id],
      ["Run", source.run_id === source.snapshot_id ? null : source.run_id],
      ["Intervals", blind ? "Independent human review" : "Model predictions"],
      ["Input SHA-256", sourceHash],
    ];
    if (!blind) {
      provenance.push(
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
    $("timeline").hidden = blind;
    $("interval-wrap").hidden = blind;
    $("focus-summary").textContent = "";
    $("focus-summary").hidden = true;
    intervals.forEach((interval) => {
      const button = document.createElement("button");
      const description = "[" + interval.start + ", " + interval.stop + ") " + label(interval);
      const fraction = (interval.stop - interval.start) / steps.length;
      button.style.width = 100 * fraction + "%";
      button.style.borderTopColor = report.colors[interval.direction || "unclassified"];
      button.textContent = fraction >= 0.09 ? label(interval) : "";
      button.title = description;
      button.setAttribute("aria-label", description);
      button.onclick = () => zoom(interval);
      $("timeline").append(button);
      const row = document.createElement("tr");
      const link = document.createElement("button");
      link.textContent = "[" + interval.start + ", " + interval.stop + ")";
      link.setAttribute("aria-label", "Zoom to " + description);
      link.onclick = () => zoom(interval);
      cell(row, "").append(link);
      const stats = interval.statistics;
      [
        label(interval),
        fmt(stats.price_change_percent),
        pct(stats.return_std),
        fmt(stats.trade_count, 0),
        pct(stats.trade_active_fraction),
        pct(stats.price_coverage),
        pct(stats.trade_count_coverage),
      ].forEach((value) => cell(row, value));
      $("intervals").append(row);
    });
    const traces = [];
    function line(name, values, panel, color, extras = {}) {
      const suffix = panel ? String(panel + 1) : "";
      traces.push({
        type: "scatter",
        mode: "lines",
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
    line("Last transaction price", observations.last_price, 0, "#0f172a");
    line("Best bid", observations.best_bid, 0, "#0891b2", {
      line: { color: "#0891b2", dash: "dot", width: 0.8 },
      visible: $("quotes").checked,
    });
    line("Best ask", observations.best_ask, 0, "#be185d", {
      line: { color: "#be185d", dash: "dash", width: 0.8 },
      visible: $("quotes").checked,
    });
    const returns = observations.last_price.map((price, index, prices) =>
      index === 0 || price == null || prices[index - 1] == null
        ? null
        : 100 * (Math.log(price) - Math.log(prices[index - 1])),
    );
    line("Adjacent log return (%)", returns, 1, "#334155");
    ["trade", "buy_order", "sell_order"].forEach((field, index) => {
      const style = {
        line: {
          color: ["#334155", "#059669", "#e11d48"][index],
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
          color: knownAbsence.map((known) => (known ? "#64748b" : "#7c3aed")),
          size: 6,
          symbol: knownAbsence.map((known) => (known ? "x-open" : "line-ns-open")),
        },
        hovertemplate: "Step %{x}<br>" + human(field) + "<br>%{customdata}<extra></extra>",
      });
    });
    const layout = {
      height: missingFields.length ? 1150 : 940,
      margin: { l: 140, r: 25, t: 30, b: 60 },
      hovermode: "x unified",
      font: { family: "system-ui, sans-serif", size: 12, color: "#334155" },
      annotations: guides.slice(0, panelCount).map((text, index) => ({
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
        fillcolor: report.colors[interval.direction || "unclassified"],
        opacity: 0.1,
        line: { width: 0 },
        layer: "below",
      })),
    };
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
        range: [-0.5, steps.length - 0.5],
        gridcolor: "#eef2f6",
        title: index === panelCount - 1 ? { text: "Step index" } : undefined,
      };
      layout["yaxis" + suffix] = {
        domain: domains[index],
        anchor: "x" + suffix,
        title: { text: title },
        gridcolor: "#eef2f6",
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
      });
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
    $("explorer").style.height = layout.height + "px";
    await Plotly.react("explorer", traces, layout, {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["toImage"],
    });
  }
  function update() {
    const generation = ++renderGeneration;
    $("render-error").hidden = true;
    $("explorer").hidden = false;
    render().catch((error) => {
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
  $("sample").onchange = () => update();
  $("experiment").onchange = () => {
    selectInstruments(samples[Number($("sample").value)].instrument_id);
    update();
  };
  $("reset").onclick = () => update();
  $("quotes").onchange = () => Plotly.restyle("explorer", { visible: $("quotes").checked }, [1, 2]);
  update();
})();
