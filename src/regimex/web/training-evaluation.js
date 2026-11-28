import {analysisStateLabel, element, launchSourceLabel, shortDate} from "./ui.js";
import {validationExample} from "./training-validation-example.js";

const number = new Intl.NumberFormat(undefined, {maximumFractionDigits: 5});
const percent = new Intl.NumberFormat(undefined, {style: "percent", maximumFractionDigits: 1});

function quantity(value, singular, plural) {
  if (!Number.isInteger(value)) return null;
  return number.format(value) + " " + (value === 1 ? singular : plural);
}

function reading(value) {
  return Number.isFinite(value) ? number.format(value) : "Unavailable";
}

function scoreEntry(label, score, bounds, level, missing = "Unavailable") {
  const entry = element("div");
  const value = element("dd", Number.isFinite(score) ? number.format(score) : missing);
  entry.append(element("dt", label), value);
  if (Number.isFinite(score) && Number.isFinite(level) && level > 0 && level < 1 &&
      Array.isArray(bounds) && bounds.length === 2 && bounds.every(Number.isFinite)) {
    const [lower, upper] = bounds;
    if (lower >= 0 && lower <= upper && upper <= 1) {
      const interval = element("small", "", "evaluation-score-interval");
      const abbreviation = element("abbr", percent.format(level) + " CI");
      abbreviation.title = percent.format(level) + " confidence interval across experiments";
      interval.append(abbreviation, " · " + number.format(lower) + "–" + number.format(upper));
      value.append(interval);
    }
  }
  return entry;
}

function showReport(content, report) {
  const split = {controlled_final_benchmark: "held-out ", fixed_model_validation: "validation "}[report.scope] || "";
  const cohort = [
    quantity(report.experiment_count, split + "experiment", split + "experiments"),
    quantity(report.instrument_sequence_count, "instrument series", "instrument series"),
    {automatic: "Automatic references", human: "Human references"}[report.reference_kind],
  ].filter(Boolean);
  const scores = element("dl", undefined, "evaluation-summary");
  const enoughExperiments = (report.confidence_intervals?.experiment_count ?? report.experiment_count) > 1;
  const uncertainty = enoughExperiments ? report.confidence_intervals : null;
  const hasPointScore = report.point_metrics?.defined !== false;
  const intervals = report.segment_metrics;
  const strict = report.strict_segment_metrics;
  const hasStrictScore = Number.isFinite(strict?.macro_f1) && Number.isFinite(intervals?.iou_threshold) &&
    strict.iou_threshold > intervals.iou_threshold && strict.iou_threshold <= 1;
  const intervalLabel = threshold => "Interval macro-F1 · " + percent.format(threshold) + " overlap";
  scores.append(scoreEntry(hasStrictScore ? intervalLabel(intervals.iou_threshold) : "Interval macro-F1",
    report.segment_macro_f1, uncertainty?.segment_macro_f1_percentile_interval, uncertainty?.confidence_level));
  if (hasStrictScore) scores.append(scoreEntry(intervalLabel(strict.iou_threshold), strict.macro_f1));
  scores.append(scoreEntry("Per-step macro-F1", hasPointScore ? report.point_macro_f1 : null,
    uncertainty?.point_macro_f1_percentile_interval, uncertainty?.confidence_level,
    hasPointScore ? "Unavailable" : "No labeled steps"));
  content.replaceChildren();
  if (cohort.length) content.append(element("p", cohort.join(" · "), "hint"));
  content.append(scores);

  if (!Array.isArray(intervals?.class_names) || !intervals.per_class) return;
  const table = element("table", undefined, "evaluation-directions");
  table.createCaption().textContent = "Intervals by direction";
  const header = table.createTHead().insertRow();
  const directions = intervals.class_names;
  for (const label of ["Metric", ...directions.map(name => name.charAt(0).toUpperCase() + name.slice(1))]) {
    const th = element("th", label);
    th.scope = "col";
    header.append(th);
  }
  const body = table.createTBody();
  for (const [label, field] of [["F1", "f1"], ["Reference", "support"], ["Predicted", "predicted"], ["Matched", "matched"]]) {
    const row = body.insertRow();
    const name = element("th", label);
    name.scope = "row";
    row.append(name);
    for (const direction of directions) {
      row.insertCell().textContent = reading(intervals.per_class[direction]?.[field]);
    }
  }
  const wrapper = element("div", undefined, "training-values");
  wrapper.append(table);
  content.append(wrapper);
  if (report.scope === "fixed_model_validation" && report.validation_examples) {
    const panel = element("div");
    const buttons = [];
    const row = body.insertRow();
    const title = element("th", "Missed");
    title.scope = "row";
    row.append(title);
    for (const direction of directions) {
      const stats = intervals.per_class[direction];
      const count = stats.support - stats.matched;
      const cell = row.insertCell();
      const example = report.validation_examples[direction];
      if (!count || !example) {
        cell.textContent = reading(count);
        continue;
      }
      const button = element("button", reading(count), "validation-example-button");
      button.type = "button";
      button.setAttribute("aria-label", `Inspect a missed ${direction} interval (${count} missed)`);
      button.setAttribute("aria-expanded", "false");
      button.addEventListener("click", () => {
        const opening = button.getAttribute("aria-expanded") === "false";
        for (const item of buttons) item.setAttribute("aria-expanded", String(opening && item === button));
        if (opening) panel.replaceChildren(validationExample(direction, example, report.reference_kind));
        else panel.replaceChildren();
      });
      buttons.push(button);
      cell.append(button);
    }
    content.append(panel);
  }
  if (Number.isFinite(intervals.iou_threshold)) {
    const threshold = percent.format(intervals.iou_threshold);
    content.append(element("p", "The table counts same-direction intervals with at least " + threshold +
      " overlap (intersection over union).", "hint"));
  }
}

function completedEvaluation(session, id) {
  const url = "/api/v1/evaluations/" + encodeURIComponent(id) + "/result";
  const result = element("div", "", "evaluation-result");
  const content = element("div");
  const status = element("p", "", "evaluation-status");
  status.setAttribute("role", "status");
  const retry = element("button", "Retry loading", "quiet");
  retry.type = "button";
  retry.hidden = true;
  const download = element("a", "Download JSON");
  download.href = url;
  download.download = "evaluation-" + id + ".json";
  result.append(status, retry, content, download);
  async function load() {
    retry.hidden = true;
    status.textContent = "Loading evaluation…";
    try {
      const report = await session.json(url);
      if (!result.isConnected) return;
      showReport(content, report);
      const fields = element("details", "", "evaluation-fields");
      const json = element("pre");
      fields.append(element("summary", "All fields"), json);
      fields.addEventListener("toggle", () => {
        if (fields.open && !json.textContent) json.textContent = JSON.stringify(report, null, 2);
      });
      content.append(fields);
      status.hidden = true;
    } catch (error) {
      if (!result.isConnected) return;
      status.textContent = error.message;
      retry.hidden = false;
    }
  }
  retry.addEventListener("click", load);
  void load();
  return result;
}

function actionRow(session, row) {
  const item = element("li");
  item.dataset.id = row.run_id;
  item.dataset.state = row.job.state;
  item.dataset.failure = row.job.failure?.message || "";
  if (row.job.kind === "evaluation" && row.job.state === "succeeded") {
    item.append(completedEvaluation(session, row.run_id));
  } else {
    const date = row.job.created_at;
    const time = element("time", shortDate(date), "evaluation-history-time");
    time.dateTime = date;
    const action = row.job.kind === "evaluation" ? "Evaluation" : "Publication";
    item.append(element("span", action + " · " + analysisStateLabel(row.job.state)), " ", time);
    if (row.job.failure?.message) item.append(element("p", row.job.failure.message));
  }
  if (row.job.launch_source) {
    item.append(element("p", "Launch source: " + launchSourceLabel(row.job.launch_source), "hint"));
  }
  return item;
}

export function showTrainingActions(list, rows, session) {
  // Polling a publication must preserve the loaded report and its keyboard focus.
  const existing = new Map(Array.from(list.children, item => [item.dataset.id, item]));
  for (const [index, row] of rows.entries()) {
    let item = existing.get(row.run_id);
    if (!item || item.dataset.state !== row.job.state || item.dataset.failure !== (row.job.failure?.message || ""))
      item = actionRow(session, row);
    if (list.children[index] !== item) list.insertBefore(item, list.children[index] || null);
  }
  while (list.children.length > rows.length) list.lastElementChild.remove();
}
