"use strict";

(() => {
  const payload = JSON.parse(document.getElementById("comparison-data").textContent);
  const evaluations = payload.evaluations;
  const first = evaluations[0];
  const number = value => value === null ? "—" : Number(value).toFixed(3);
  const percent = value => value === null ? "—" : `${(100 * value).toFixed(1)}%`;
  const table = (id, headings, rows) => {
    const element = document.createElement("table");
    const head = element.createTHead().insertRow();
    headings.forEach(text => {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = text;
      head.append(cell);
    });
    const body = element.createTBody();
    rows.forEach(values => {
      const row = body.insertRow();
      values.forEach(value => {
        const cell = row.insertCell();
        if (value instanceof Node) cell.append(value);
        else cell.textContent = String(value);
      });
    });
    document.getElementById(id).replaceChildren(element);
  };
  const score = (value, interval) => {
    const element = document.createElement("span");
    element.textContent = number(value);
    if (value !== null) {
      const range = document.createElement("span");
      range.className = "range";
      range.textContent = `[${number(interval[0])}, ${number(interval[1])}]`;
      element.append(range);
    }
    return element;
  };
  document.getElementById("population").textContent = `${first.experiment_count} experiments · ${first.instrument_sequence_count} instrument series · ${first.scope === "fixed_model_validation" ? "validation" : "final benchmark"}`;
  document.getElementById("reference-note").textContent = first.reference_kind === "automatic"
    ? `Agreement with automatic labels (${first.reference_policy}). These scores do not measure independent human accuracy.`
    : `Agreement with attributed human references (${first.reference_policy}).`;
  document.getElementById("metric-note").textContent = `Higher F1 is better. Interval IoU ≥ 0.50; boundary tolerance ${first.boundary_metrics.tolerance_steps} steps. Typed boundaries also require both directions to agree. Brackets: 95% whole-experiment intervals. Lag is prediction − reference among aligned typed transitions, without a distance cutoff; positive means later. Distant pairs can represent different events.`;
  table("overview", ["Model", "Interval macro-F1 ↑", "Step macro-F1 ↑", "Boundary timing F1 ↑", "Typed boundary F1 ↑", "Median typed lag, steps"], evaluations.map(item => [
    item.model_version_id,
    score(item.segment_macro_f1, item.confidence_intervals.segment_macro_f1_percentile_interval),
    score(item.point_macro_f1, item.confidence_intervals.point_macro_f1_percentile_interval),
    score(item.boundary_metrics.defined ? item.boundary_metrics.f1 : null, item.confidence_intervals.boundary_f1_percentile_interval),
    score(item.typed_boundary_metrics.defined ? item.typed_boundary_metrics.f1 : null, item.confidence_intervals.typed_boundary_f1_percentile_interval),
    number(item.typed_boundary_metrics.alignment.signed_lag_steps.median_steps)
  ]));
  const paired = () => {
    const metric = document.getElementById("paired-metric").value;
    table("paired", ["Model", `Δ vs ${first.model_version_id}`, "Experiments ↑", "Tied", "Experiments ↓"], payload.paired_comparisons.map(item => {
      const values = item.metrics[metric];
      return [item.candidate_model_version_id, score(values.delta, values.percentile_interval), values.improved_experiments, values.tied_experiments, values.worse_experiments];
    }));
  };
  document.getElementById("paired-metric").addEventListener("change", paired);
  paired();
  const selectionNote = first.scope === "fixed_model_validation"
    ? "Reused selection data limits conclusions about unseen experiments."
    : "Models were fixed before opening this final benchmark; these data were not used for selection.";
  document.getElementById("paired-note").textContent = `Candidate − ${first.model_version_id}; 95% paired intervals use the same ${first.confidence_intervals.repetitions} experiment draws for both models. Counts compare individual experiment scores; overall Δ pools counts. ${selectionNote} Intervals condition on the saved models and do not include training-seed variation.`;
  const directions = () => {
    const metric = document.getElementById("metric-kind").value;
    table("directions", ["Model / direction", "Precision", "Recall", "F1", "Reference count", "Predicted count", "Matched"], evaluations.flatMap(item => first.segment_metrics.class_names.map(direction => {
      const values = item[metric].per_class[direction];
      return [`${item.model_version_id} / ${direction}`, number(values.precision), number(values.recall), number(values.f1), values.support, values.predicted, values.matched];
    })));
  };
  document.getElementById("metric-kind").addEventListener("change", directions);
  directions();
  const lengthRow = (label, values, ratio, coverage) => [label, values.interval_count, values.minimum_steps ?? "—", number(values.median_steps), values.maximum_steps ?? "—", number(ratio), percent(coverage)];
  table("durations", ["Model", "Intervals", "Min steps", "Median steps", "Max steps", "Count / reference", "Eligible coverage"], [
    lengthRow("Reference", first.fragmentation.reference, first.fragmentation.reference.interval_count ? 1 : null, first.coverage.reference_classified_step_count ? 1 : null),
    ...evaluations.map(item => lengthRow(item.model_version_id, item.fragmentation.predicted, item.fragmentation.predicted_to_reference_interval_ratio, item.coverage.eligible_prediction_coverage))
  ]);
  const selector = document.getElementById("experiment");
  const experiments = [...first.experiment_statistics].sort((a, b) => a.experiment_id.localeCompare(b.experiment_id));
  experiments.forEach((item, index) => {
    const option = document.createElement("option");
    option.value = item.experiment_id;
    option.textContent = `${index + 1}. ${item.experiment_id.slice(0, 16)}`;
    selector.append(option);
  });
  const experiment = () => table("experiment-metrics", ["Model", "Interval macro-F1", "Step macro-F1", "Boundary timing F1", "Typed boundary F1"], evaluations.map(item => {
    const entry = item.experiment_statistics.find(value => value.experiment_id === selector.value);
    const hasBoundaries = entry.boundary_counts.some(value => value > 0);
    const hasTyped = entry.typed_boundary_counts.some(value => value > 0);
    return [item.model_version_id, number(entry.segment_macro_f1), number(entry.point_macro_f1), hasBoundaries ? number(entry.boundary_f1) : "—", hasTyped ? number(entry.typed_boundary_f1) : "—"];
  }));
  selector.addEventListener("change", experiment);
  experiment();
  const provenance = document.getElementById("provenance");
  const entries = [
    ["Report", payload.version], ["Dataset SHA-256", first.dataset_sha256],
    ["References SHA-256", first.reference_sha256], ["Analysis profile SHA-256", first.analysis_profile_sha256],
    ["Decoder", first.decoder_version], ["Bootstrap seed", first.confidence_intervals.seed],
    ...evaluations.flatMap((item, index) => [
      [item.model_version_id, `Bundle ${item.bundle_sha256}`],
      ["Evaluation SHA-256", payload.evaluation_sha256[index]],
      ["Training reference", `${item.training_reference_kind} / ${item.training_reference_policy} / ${item.training_reference_sha256}`],
      ["Selection references SHA-256", item.validation_selection_reference_sha256],
      ["Inference runtime", JSON.stringify(item.inference_runtime)]
    ]),
    ["Experiments", "Every evaluated experiment is included, in sorted identity order."]
  ];
  const list = document.createElement("dl");
  entries.forEach(([name, value]) => {
    const term = document.createElement("dt");
    const description = document.createElement("dd");
    term.textContent = name;
    description.textContent = value;
    list.append(term, description);
  });
  provenance.append(list);
})();
