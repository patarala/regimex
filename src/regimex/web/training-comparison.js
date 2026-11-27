import {$, analysisStateLabel, element, shortDate} from "./ui.js";
import {finished} from "./training-archive.js";
import {isTransientRequestError} from "./session.js";
import {architectureName} from "./training-architectures.js";
import {differentSettings} from "./training-json.js";
import {isImported} from "./training-import.js";

function sameValidation(left, right) {
  if (left.analysis_profile_id !== right.analysis_profile_id || !left.analysis_profile_id) return false;
  if (left.origin === "external" || right.origin === "external") {
    if (left.origin !== right.origin) return false;
    const a = left.evidence, b = right.evidence;
    const aValidation = a.validation_dataset_sha256 || a.dataset_sha256;
    const bValidation = b.validation_dataset_sha256 || b.dataset_sha256;
    // The split manifest describes the primary corpus. A separate validation dataset
    // binds its own partitions in its content digest.
    const bothUsePrimary = aValidation === a.dataset_sha256 && bValidation === b.dataset_sha256;
    return aValidation === bValidation
      && (!bothUsePrimary || a.split_manifest_sha256 === b.split_manifest_sha256)
      && a.validation_references.content_sha256 === b.validation_references.content_sha256
      && a.validation_references.direction_semantics_sha256 === b.validation_references.direction_semantics_sha256;
  }
  return ["dataset_id", "validation_reference_set_id"].every(key => left[key] && left[key] === right[key]);
}

function setText(node, value) {
  if (node.textContent !== value) node.textContent = value;
}

function table(caption, headings) {
  const result = element("table");
  result.append(element("caption", caption, "visually-hidden"));
  const header = result.createTHead().insertRow();
  for (const heading of [null, ...headings]) {
    const cell = element("th");
    cell.scope = "col";
    if (heading) {
      const link = element("a", heading.name);
      link.href = heading.href;
      link.dataset.workspaceLink = "";
      cell.append(link);
    }
    header.append(cell);
  }
  result.createTBody();
  return result;
}

function updateRows(result, rows) {
  const body = result.tBodies[0];
  rows.forEach(([label, left, right], index) => {
    let row = body.rows[index];
    if (!row) {
      row = body.insertRow();
      const name = element("th");
      name.scope = "row";
      row.append(name, element("td"), element("td"));
    }
    [label, left, right].forEach((value, column) => setText(row.cells[column], value));
  });
  while (body.rows.length > rows.length) body.deleteRow(-1);
}

const pollInterval = 2500;

export function createTrainingComparison({readRun, session, runPath, duration, onStateChange}) {
  const panel = $("training-comparison"), status = $("training-comparison-status");
  const content = $("training-comparison-results"), retry = $("training-comparison-retry");
  let current = null, timer;

  function close() {
    clearTimeout(timer);
    current?.request?.abort();
    current = null;
    panel.hidden = true;
    content.replaceChildren();
    status.textContent = "";
    retry.hidden = true;
  }

  function runHeadings(runs) {
    const names = runs.map(run => run.display_name || (isImported(run) ? run.training.model_version_id
      : architectureName(run.training.architecture_id, true)));
    return runs.map((run, index) => ({
      name: (index === 0 ? "A · " : "B · ") + names[index] + (names[0] === names[1] ? " #" + run.run_id.slice(-8) : ""),
      href: runPath(run.run_id),
    }));
  }

  function updateHeadings() {
    if (!current?.overview) return;
    const headings = runHeadings(current.runs);
    for (const result of [current.overview, current.settings]) {
      if (!result) continue;
      result.tHead.querySelectorAll("a").forEach((link, index) => {
        const {name, href} = headings[index];
        setText(link, name);
        if (link.getAttribute("href") !== href) link.setAttribute("href", href);
      });
    }
  }

  function render(view, runs) {
    const row = (name, select) => [name, ...runs.map(run => select(run) ?? "—")];
    const rows = [
      row("Status", r => isImported(r) ? "Imported" : analysisStateLabel(r.job.state)),
      row(runs.some(isImported) ? "Recorded" : "Created", r => shortDate(r.job.created_at)),
      row("Dataset", r => isImported(r) ? r.training.evidence.dataset_sha256 : r.training.dataset_id),
      row("Architecture", r => architectureName(r.training.architecture_id, true)),
      row("Trainable parameters", r => r.trainable_parameters?.toLocaleString()),
      ...[["Elapsed", "elapsed_seconds"], ["In queue", "queued_seconds"],
        ["With worker", "running_seconds"], ["Model fit time", "fit_seconds"]].map(([name, key]) =>
        row(name, r => r.statistics?.[key] == null || (isImported(r) && key !== "fit_seconds")
          ? null : duration(r.statistics[key]))),
    ];
    if (view.scores) rows.push(["Validation interval macro-F1", ...view.scores.map(value => value.toFixed(4))]);
    if (!view.overview) {
      const headings = runHeadings(runs);
      view.overview = table("Run results", headings);
      content.append(view.overview);
      if (isImported(runs[0]) === isImported(runs[1]) && runs.every(run => run.configuration)) {
        const differences = differentSettings(runs[0].configuration, runs[1].configuration);
        content.append(element("h3", "Settings that differ"));
        if (differences.length) {
          view.settings = table("Settings that differ", headings);
          content.append(view.settings);
          updateRows(view.settings, differences);
        } else content.append(element("p", "Both runs used the same settings.", "hint"));
      }
    } else updateHeadings();
    updateRows(view.overview, rows);
  }

  async function refresh(poll = false) {
    clearTimeout(timer);
    const view = current;
    if (!view || view.request || document.hidden) return;
    const request = new AbortController();
    view.request = request;
    const timeout = setTimeout(() => request.abort(), 15000);
    let delay = null;
    try {
      const runs = await Promise.all(view.ids.map((id, i) =>
        poll && view.runs && finished(view.runs[i].job) ? view.runs[i] : readRun(id, {signal: request.signal})));
      if (current !== view || document.hidden) return;
      if (poll && view.runs?.some((run, i) => run.job.state !== runs[i].job.state)) onStateChange();
      view.runs = runs;
      retry.hidden = true;
      render(view, runs);
      if (!runs.every(run => run.job.state === "succeeded")) {
        setText(status, "Both runs must succeed to compare their validation scores.");
        if (runs.some(run => !finished(run.job))) delay = pollInterval;
      } else if (!sameValidation(runs[0].training, runs[1].training)) {
        setText(status, "Comparing validation scores requires the same dataset, validation labels and analysis profile.");
      } else {
        if (!view.scores) {
          setText(status, "Loading validation scores…");
          const scores = await Promise.all(runs.map(async run => {
            if (run.output?.validation_segment_macro_f1 != null) return run.output.validation_segment_macro_f1;
            const path = "/api/v1/training-runs/" + encodeURIComponent(run.run_id) + "/validation-score";
            return (await session.json(path, {signal: request.signal})).validation_segment_macro_f1;
          }));
          if (current !== view || document.hidden) return;
          view.scores = scores;
          render(view, runs);
        }
        const change = view.scores[1] - view.scores[0];
        setText(status, "Validation interval macro-F1 (B − A): " + (change > 0 ? "+" : "") + change.toFixed(4));
      }
      view.retryDelay = pollInterval;
    } catch (error) {
      request.abort();
      if (current !== view || document.hidden) return;
      setText(status, error.message);
      retry.hidden = false;
      if (isTransientRequestError(error)) {
        delay = view.retryDelay;
        view.retryDelay = Math.min(delay * 2, 30000);
      }
    } finally {
      clearTimeout(timeout);
      view.request = null;
      if (current === view && !document.hidden && delay !== null) timer = setTimeout(() => refresh(true), delay);
    }
  }

  function show(ids) {
    close();
    panel.hidden = false;
    $("training-comparison-title").focus();
    if (ids.length !== 2 || ids[0] === ids[1]) {
      status.textContent = "Select two different runs in the history and choose Compare.";
      return;
    }
    current = {ids, retryDelay: pollInterval};
    status.textContent = "Loading runs…";
    return refresh();
  }
  retry.addEventListener("click", () => refresh());
  return {close, show, refresh, updateHeadings};
}
