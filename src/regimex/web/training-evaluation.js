import {analysisStateLabel, element, shortDate} from "./ui.js";

const number = new Intl.NumberFormat(undefined, {maximumFractionDigits: 5});
const percent = new Intl.NumberFormat(undefined, {style: "percent", maximumFractionDigits: 1});

function quantity(value, singular, plural) {
  if (!Number.isInteger(value)) return null;
  return number.format(value) + " " + (value === 1 ? singular : plural);
}

function reading(value) {
  return Number.isFinite(value) ? number.format(value) : "Unavailable";
}

function showReport(content, report) {
  const cohort = [
    quantity(report.experiment_count, "validation experiment", "validation experiments"),
    quantity(report.instrument_sequence_count, "instrument series", "instrument series"),
    {automatic: "Automatic references", human: "Human references"}[report.reference_kind],
  ].filter(Boolean);
  const scores = element("dl", undefined, "evaluation-summary");
  for (const [label, value] of [
    ["Interval macro-F1", reading(report.segment_macro_f1)],
    ["Per-step macro-F1", report.point_metrics?.defined === false ? "No labeled steps" : reading(report.point_macro_f1)],
  ]) {
    const pair = element("div");
    pair.append(element("dt", label), element("dd", value));
    scores.append(pair);
  }
  content.replaceChildren();
  if (cohort.length) content.append(element("p", cohort.join(" · "), "hint"));
  content.append(scores);

  const intervals = report.segment_metrics;
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
  if (Number.isFinite(intervals.iou_threshold)) {
    const threshold = percent.format(intervals.iou_threshold);
    content.append(element("p", "A match requires the same direction and at least " + threshold +
      " overlap (intersection over union).", "hint"));
  }
}

function completedEvaluation(session, id, time) {
  const url = "/api/v1/evaluations/" + encodeURIComponent(id) + "/result";
  const details = element("details", undefined, "evaluation-result");
  const summary = element("summary");
  summary.append(element("span", "View evaluation"), " ", time);
  const content = element("div");
  const status = element("p", "", "evaluation-status");
  status.setAttribute("role", "status");
  status.hidden = true;
  const retry = element("button", "Retry loading", "quiet");
  retry.type = "button";
  retry.hidden = true;
  const download = element("a", "Download JSON");
  download.href = url;
  download.download = "evaluation-" + id + ".json";
  details.append(summary, status, retry, content, download);
  let loaded = false, loading = false;
  async function load() {
    if (loaded || loading || !details.open) return;
    loading = true;
    retry.hidden = true;
    status.hidden = false;
    status.textContent = "Loading evaluation…";
    try {
      const report = await session.json(url);
      if (!details.isConnected) return;
      showReport(content, report);
      const fields = element("details", "", "evaluation-fields");
      const json = element("pre");
      fields.append(element("summary", "All fields"), json);
      fields.addEventListener("toggle", () => {
        if (fields.open && !json.textContent) json.textContent = JSON.stringify(report, null, 2);
      });
      content.append(fields);
      loaded = true;
      status.hidden = true;
    } catch (error) {
      if (!details.isConnected) return;
      status.textContent = error.message;
      retry.hidden = false;
    } finally {
      loading = false;
    }
  }
  retry.addEventListener("click", load);
  details.addEventListener("toggle", load);
  return details;
}

function actionRow(session, row) {
  const item = element("li");
  item.dataset.id = row.run_id;
  item.dataset.state = row.job.state;
  item.dataset.failure = row.job.failure?.message || "";
  const date = row.job.created_at;
  const time = element("time", shortDate(date), "evaluation-history-time");
  time.dateTime = date;
  if (row.job.kind === "evaluation" && row.job.state === "succeeded") {
    item.append(completedEvaluation(session, row.run_id, time));
  } else {
    const action = row.job.kind === "evaluation" ? "Evaluation" : "Publication";
    item.append(element("span", action + " · " + analysisStateLabel(row.job.state)), " ", time);
    if (row.job.failure?.message) item.append(element("p", row.job.failure.message));
  }
  return item;
}

export function showTrainingActions(list, rows, session) {
  // Keep an open evaluation and its keyboard focus when another job changes state.
  const existing = new Map(Array.from(list.children, item => [item.dataset.id, item]));
  for (const [index, row] of rows.entries()) {
    let item = existing.get(row.run_id);
    if (!item || item.dataset.state !== row.job.state || item.dataset.failure !== (row.job.failure?.message || ""))
      item = actionRow(session, row);
    if (list.children[index] !== item) list.insertBefore(item, list.children[index] || null);
  }
  while (list.children.length > rows.length) list.lastElementChild.remove();
}
