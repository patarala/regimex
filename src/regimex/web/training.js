import {$, analysisStateLabel, bindTooltip, element, launchSourceLabel, shortDate} from "./ui.js";
import {showTrainingActions} from "./training-evaluation.js";
import {createFinalBenchmark} from "./training-benchmark.js";
import {createTrainingMetrics} from "./training-metrics.js";
import {createTrainingProgress} from "./training-progress.js";
import {createRunArchive, finished} from "./training-archive.js";
import {architectureName, createTrainingForm, datasetName, shortId} from "./training-form.js";
import {readTrainingRuns, renamedConfiguration, trainingRequestKey} from "./training-json.js";
import {createTrainingComparison} from "./training-comparison.js";
import {createRunNameEditor} from "./training-name.js";
import {parameterCountLabel} from "./training-size.js";
import {queuedStageLabel} from "./queue-status.js";
import {isTransientRequestError} from "./session.js";
import {importedFacts, isImported, showImportedResults} from "./training-import.js";

const api = "/api/v1/";
const pageSize = 12;
const inProgress = (job) => ["queued", "running", "cancel_requested"].includes(job.state);
const route = (id) => api + "training-runs/" + encodeURIComponent(id);

function currentEvaluation(rows) {
  // Actions are newest first. Older installations can have several successful evaluations.
  return rows.findLast(row => row.job.state === "succeeded") ||
    rows.findLast(row => inProgress(row.job)) || rows[0];
}

function duration(seconds) {
  if (seconds > 0 && seconds < 1) return "<1 s";
  const whole = Math.floor(seconds);
  if (whole < 60) return whole + " s";
  const minutes = Math.floor(whole / 60);
  if (minutes < 60) return minutes + " min " + whole % 60 + " s";
  return Math.floor(minutes / 60) + " h " + minutes % 60 + " min";
}

function setText(node, value) {
  if (node.textContent !== value) node.textContent = value;
}

function showTiming(statistics) {
  $("training-timing").hidden = !statistics;
  if (!statistics) return;
  for (const [id, seconds] of [["training-elapsed", statistics.elapsed_seconds],
    ["training-queued", statistics.queued_seconds], ["training-running", statistics.running_seconds]]) {
    const value = $(id);
    value.parentElement.hidden = seconds == null;
    if (seconds != null) setText(value, duration(seconds));
  }
}

function runContext(row) {
  return [architectureName(row.training.architecture_id, true),
    row.trainable_parameters && parameterCountLabel(row.trainable_parameters),
    isImported(row) ? row.training.model_version_id : datasetName(row.training.dataset_id)].filter(Boolean).join(" · ");
}

function historyDate(value) {
  const date = new Date(value);
  return date.toLocaleString(undefined, {
    month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
    year: date.getFullYear() === new Date().getFullYear() ? undefined : "numeric",
  });
}

function historyEntry(row, view) {
  const {architecture_id, dataset_id} = row.training;
  const imported = isImported(row), model = architectureName(architecture_id, true);
  const key = imported ? null : shortId(dataset_id);
  return {
    id: row.run_id, name: row.display_name || (imported ? row.training.model_version_id : model + " · " + datasetName(dataset_id)),
    compactTitle: !imported && !row.display_name && key !== dataset_id ? {model, key} : null,
    created: row.job.created_at, date: historyDate(row.job.created_at), state: imported ? "imported" : row.job.state,
    archivedAt: row.archived_at, showArchiveIcon: view === "all" && Boolean(row.archived_at),
    architecture: architectureName(architecture_id), dataset: dataset_id, parameters: row.trainable_parameters,
    version: imported ? row.training.model_version_id : null,
  };
}

function historyTitle(row) {
  const heading = element("span", "", "training-run-title");
  if (row.showArchiveIcon) heading.append($("training-archive-icon").content.cloneNode(true));
  const title = element("strong");
  title.dir = "auto";
  if (row.compactTitle) {
    title.className = "training-dataset-title";
    title.append(row.compactTitle.model, element("span", " · Dataset ", "visually-hidden"),
      element("span", row.compactTitle.key, "training-dataset-key"));
  } else {
    const [first, ...parts] = row.name.split("_");
    title.append(first);
    for (const part of parts) title.append("_", document.createElement("wbr"), part);
  }
  heading.append(title);
  return heading;
}

function appendRunInfo(item, button, row, label) {
  const info = element("button", "", "field-help-toggle training-run-info");
  info.type = "button";
  info.dataset.runInfo = row.id;
  info.setAttribute("aria-label", "Details for " + label);
  const icon = element("span", "i");
  icon.setAttribute("aria-hidden", "true");
  info.append(icon);
  const note = element("div", "", "field-help run-info");
  const model = element("div", "", "run-info-line");
  model.append(element("strong", row.architecture));
  if (row.parameters != null) {
    model.append(" ", element("span", parameterCountLabel(row.parameters), "run-info-size"));
  }
  const dataset = element("div", "", "run-info-line");
  dataset.append(element("span", row.version ? "Model version:" : "Dataset:"), " ",
    element("span", row.version || row.dataset, "run-info-dataset"));
  note.append(model, dataset, element("div", (row.version ? "Imported " : "Created ") + shortDate(row.created)));
  if (row.archivedAt) note.append(element("div", "Archived " + shortDate(row.archivedAt)));
  note.id = "run-info-" + row.id;
  item.append(info, note);
  bindTooltip(info, note, {trigger: button});
}

// Delayed list and detail responses can predate a rename or weight deletion.
function withSavedChanges(row, known) {
  if (known?.weights_deletion_id && !row.weights_deletion_id)
    row = {...row, weights_deletion_id: known.weights_deletion_id};
  if ((known?.name_revision ?? 0) > (row.name_revision ?? 0)) {
    row = {...row, display_name: known.display_name, name_revision: known.name_revision};
    if (row.configuration && !isImported(row)) row.configuration = renamedConfiguration(row.configuration, row.display_name);
  }
  return row;
}

export function createTraining({session, navigate, deletion, onPublication}) {
  let opened = false, epoch = 0, selection = 0, listRequest = 0;
  let offset = 0, timer = null, busy = false, pollRevision = 0;
  let run = null, evaluation = null, activeModel, metricsLoaded = false;
  let pendingDownstream = false, retryDelay = 2500, target = null;
  let renderedFacts = null;
  let renderedEvaluations = null;
  let draftRevision = 0, submittingDraft = null, acceptedDraftRun = null;
  let editedDraft = false;
  let draftRequests = new Map();
  let view = "active", listedView = null, listedOffset = 0, renderedRows = null;
  let listed = new Map();
  const form = createTrainingForm({session, eachPage});
  const names = createRunNameEditor({session, getRun: () => run, applyName});
  const metrics = createTrainingMetrics();
  const progress = createTrainingProgress(session, metrics);
  const benchmark = createFinalBenchmark({session, action, refresh: refreshDetail});
  let comparedIds = [];
  const archive = createRunArchive({apply: setArchived, renderRows,
    compare: ids => navigate(runPath("compare", view, ids))});
  const runsToggle = $("training-runs-toggle");
  function runPath(id, filter = view, ids = comparedIds) {
    const query = new URLSearchParams();
    if (filter !== "active") query.set("view", filter);
    if (id === "compare" && ids.length === 2) {
      query.set("left", ids[0]);
      query.set("right", ids[1]);
    }
    return "/training" + (id ? "/" + encodeURIComponent(id) : "") + (query.size ? "?" + query : "");
  }
  const comparison = createTrainingComparison({readRun, session, runPath, duration});
  const pending = new Map();
  const isAdmin = () => session.value?.principal.role === "admin";
  const current = (stamp) => opened && stamp === epoch && isAdmin();
  const selected = (stamp) => opened && stamp === selection && isAdmin();

  async function readRecords(path, options) {
    const account = session.epoch;
    const response = await session.request(path, options);
    const text = await response.text();
    if (session.epoch !== account) throw new Error("The workspace changed. Refresh the page to continue.");
    return readTrainingRuns(text);
  }

  function readRun(id) { return readRecords(route(id)); }

  function message(value = "", error = false) {
    $("training-notice").textContent = value;
    $("training-notice").classList.toggle("error", error);
    $("training-notice").hidden = !value;
  }

  function markCurrent(id) {
    for (const item of $("training-runs").children) {
      const button = item.querySelector("[data-run-id]");
      button.setAttribute("aria-current", String(button.dataset.runId === id));
    }
  }

  function deselect() {
    selection++;
    clearTimeout(timer);
    form.close();
    names.close();
    retryDelay = 2500;
    $("training-sync").hidden = true;
    $("training-metric-error").hidden = true;
    $("training-delete-weights").hidden = true;
    $("training-run-details").open = false;
    markCurrent(null);
    run = evaluation = null;
    archive.showRun(null);
    activeModel = undefined;
    pendingDownstream = false;
    metricsLoaded = false;
    progress.stop();
    metrics.clear();
    comparison.close();
    $("training-downstream").hidden = $("training-detail").hidden = true;
    for (const id of ["training-state", "training-stage", "training-settings",
      "training-publication"]) $(id).textContent = "";
    $("training-detail-title").textContent = "Run";
    $("training-metric-data").removeAttribute("href");
    $("training-metric-data").removeAttribute("download");
    $("training-metric-data").hidden = true;
    $("training-facts").replaceChildren();
    renderedFacts = null;
    renderedEvaluations = null;
    benchmark.reset();
    $("training-external-evaluations").hidden = true;
    $("training-imported-results").replaceChildren();
    $("training-import-data").removeAttribute("href");
    $("training-import-data").removeAttribute("download");
    $("training-import-data").hidden = true;
    showTiming(null);
    $("training-actions").replaceChildren();
    $("training-publications").replaceChildren();
  }

  function close() {
    epoch++;
    opened = false;
    target = null;
    listed.clear();
    listedView = renderedRows = null;
    archive.reset();
    busy = false;
    deselect();
    $("training").hidden = true;
    $("training-config").hidden = true;
    $("training-runs").replaceChildren();
    $("training-runs").setAttribute("aria-busy", "false");
    $("training-runs").inert = false;
    message();
  }

  function reset() {
    close();
    submittingDraft = acceptedDraftRun = null;
    discardDraft();
    pending.clear();
    $("open-training").hidden = true;
  }

  function reportIfCurrent(stamp, error) {
    if (current(stamp)) message(error.message, true);
  }

  function renderRows(reuse = false) {
    const focused = document.activeElement?.dataset;
    const page = [...listed.values()];
    const rows = page.map(row => historyEntry(row, listedView));
    archive.showPage(page, listedView !== view);
    const content = JSON.stringify([archive.selecting, rows]);
    if (reuse && renderedRows === content) return;
    renderedRows = content;
    $("training-runs").replaceChildren(...rows.map(row => {
      const item = document.createElement("li");
      const button = element("button", "");
      button.type = "button";
      button.className = "quiet training-run-link";
      button.dataset.runId = row.id;
      button.setAttribute("aria-current", String(row.id === target));
      const time = element("time", row.date);
      time.dateTime = row.created;
      const sameLabel = rows.some(other => other !== row && other.name === row.name && other.date === row.date);
      button.append(historyTitle(row));
      const state = element("span", row.state === "imported" ? "Imported" : analysisStateLabel(row.state));
      state.dataset.state = row.state;
      const meta = element("span", "", "training-run-meta");
      meta.append(time, state);
      const suffix = sameLabel ? "#" + row.id.slice(-8) : "";
      if (suffix) meta.append(element("span", suffix, "training-run-id"));
      button.append(meta);
      button.addEventListener("click", () => chooseFromHistory(row.id));
      const label = suffix ? row.name + " " + suffix : row.name;
      const check = archive.checkbox(listed.get(row.id), label);
      if (check) item.append(check);
      item.append(button);
      appendRunInfo(item, button, row, label);
      return item;
    }));
    if (focused?.runId || focused?.selectRun || focused?.runInfo)
      for (const control of $("training-runs").querySelectorAll("button, input"))
        if ((focused.runId && control.dataset.runId === focused.runId) ||
            (focused.selectRun && control.dataset.selectRun === focused.selectRun) ||
            (focused.runInfo && control.dataset.runInfo === focused.runInfo)) control.focus({preventScroll: true});
  }

  async function list(preselect = false) {
    const stamp = epoch, request = ++listRequest;
    const valid = () => current(stamp) && request === listRequest;
    const history = $("training-runs"), empty = $("training-empty");
    const prev = $("training-prev"), next = $("training-next");
    history.setAttribute("aria-busy", "true");
    history.inert = listedView !== view;
    prev.disabled = next.disabled = true;
    archive.showPage([...listed.values()], history.inert);
    if (history.inert) empty.hidden = true;
    let page;
    try {
      page = await readRecords(api + "training-runs?archive=" + view + "&limit=" + (pageSize + 1) + "&offset=" + offset);
    } catch (error) {
      if (valid()) {
        history.setAttribute("aria-busy", "false");
        prev.disabled = next.disabled = history.inert;
        if (!history.inert) offset = listedOffset;
        empty.textContent = (history.inert && listed.size ? "Showing previous results. " : "") + error.message;
        empty.hidden = false;
      }
      return;
    }
    if (!valid()) return;
    if (!page.items.length && offset > 0) {
      offset = Math.max(0, offset - pageSize);
      return list(preselect);
    }
    const rows = page.items.slice(0, pageSize).map(row =>
      withSavedChanges(row, run?.run_id === row.run_id ? run : listed.get(row.run_id)));
    listedView = view;
    listedOffset = offset;
    listed = new Map(rows.map(row => [row.run_id, row]));
    history.inert = false;
    renderRows(true);
    history.setAttribute("aria-busy", "false");
    if (run && listed.has(run.run_id)) {
      const item = listed.get(run.run_id);
      const updated = withSavedChanges(run, item);
      if (updated !== run) {
        run = updated;
        showTitle();
        if (run.weights_deletion_id) showRemovedWeights();
      }
      if ((item.archived_at ?? null) !== (run.archived_at ?? null)) void refreshDetail();
    }
    empty.hidden = page.items.length !== 0;
    empty.textContent = {active: "No current runs.", archived: "No archived runs.", all: "No training runs yet."}[view];
    prev.disabled = next.disabled = false;
    prev.hidden = offset === 0;
    next.hidden = page.items.length <= pageSize;
    if (preselect && target === null && rows.length) navigate(runPath(rows[0].run_id), true);
  }

  // A network error can arrive after the server committed. Reuse that exact request's key.
  async function post(path, body, requests = pending) {
    const request = JSON.stringify([path, trainingRequestKey(body)]);
    const key = requests.get(request) || crypto.randomUUID();
    requests.set(request, key);
    const options = {method: "POST", body, headers: {"Idempotency-Key": key}};
    const result = path === api + "training-runs" ? await readRecords(path, options) : await session.json(path, options);
    // A draft can be edited while its request is in flight, then restored to the same settings.
    if (requests === pending && requests.get(request) === key) requests.delete(request);
    return result;
  }

  async function eachPage(path, isCurrent) {
    const rows = [];
    const join = path.includes("?") ? "&" : "?";
    while (isCurrent()) {
      const page = await session.json(path + join + "limit=100&offset=" + rows.length);
      if (!isCurrent()) break;
      rows.push(...page.items);
      if (page.items.length < 100) break;
    }
    return rows;
  }

  function showTitle() {
    const context = runContext(run);
    setText($("training-detail-title"), run.display_name || context);
    const caption = $("training-created");
    const prefix = run.display_name ? context : "";
    const date = (isImported(run) ? "Imported " : "Created ") + shortDate(run.job.created_at);
    if (caption.lastElementChild?.dateTime !== run.job.created_at || caption.textContent !== prefix + date) {
      const created = element("time", date);
      created.dateTime = run.job.created_at;
      caption.replaceChildren(prefix, created);
    }
    $("training-archived").hidden = !run.archived_at;
    archive.showRun(run);
  }

  function applyName(id, name, revision) {
    const saved = {display_name: name, name_revision: revision};
    if (run?.run_id === id) {
      run = withSavedChanges(run, saved);
      showTitle();
      setText($("training-settings"), run.configuration);
    }
    if (listed.has(id)) {
      listed.set(id, withSavedChanges(listed.get(id), saved));
      renderRows();
    }
  }

  function showRemovedWeights() {
    $("training-downstream").hidden = false;
    $("training-evaluate").hidden = $("training-publish").hidden = $("training-delete-weights").hidden = true;
    $("training-final-next").hidden = true;
    $("training-deletion-receipt").hidden = false;
    $("training-publication").textContent = "Model weights removed. This version can no longer be evaluated or published.";
    $("training-publication").hidden = false;
  }

  function weightsRemoved(runId, deletionId) {
    const saved = {weights_deletion_id: deletionId};
    if (listed.has(runId)) listed.set(runId, withSavedChanges(listed.get(runId), saved));
    if (run?.run_id === runId) {
      run = withSavedChanges(run, saved);
      showRemovedWeights();
    }
    if (opened) void reloadArchivedRuns([runId]);
  }

  async function reloadArchivedRuns(ids) {
    const detail = ids.includes(target) ? refreshDetail() : null;
    await Promise.all([detail, list()]);
  }

  function refreshDetail() {
    benchmark.invalidate();
    if (target === "compare") return comparison.show(comparedIds, false);
    if (run) {
      // Verified evidence can be attached after a run finishes.
      if (run.job.state === "succeeded") metricsLoaded = false;
      return refresh(selection);
    }
    if (target && target !== "new") {
      // Supersede a detail request started before the mutation was acknowledged.
      return loadRun(target, ++selection);
    }
  }

  async function setArchived(ids, archived) {
    const account = session.epoch, workspace = epoch;
    let saved = false;
    try {
      await session.request(api + "training-runs/archive", {method: "PATCH", body: {run_ids: ids, archived}});
      saved = true;
      if (current(workspace))
        message(ids.length + (ids.length === 1 ? " run " : " runs ") + (archived ? "archived." : "restored."));
    } catch (error) {
      reportIfCurrent(workspace, error);
    }
    // A lost confirmation can follow a committed change. Read the current
    // view rather than guessing which rows still belong in it.
    if (opened && session.epoch === account) await reloadArchivedRuns(ids);
    return saved;
  }

  function facts() {
    const spec = run.training;
    const stats = run.statistics;
    showTitle();
    if (run.weights_deletion_id) showRemovedWeights();
    const imported = isImported(run);
    setText($("training-state"), imported ? "Imported" : analysisStateLabel(run.job.state));
    $("training-state").dataset.state = imported ? "imported" : run.job.state;
    $("training-copy-run").hidden = imported;
    setText($("training-details-label"), imported ? "Bundle, weights and settings" : "Run details and settings");
    showTiming(imported ? null : stats);
    const entries = imported ? importedFacts(run) : [["Run", run.run_id], ["Dataset", spec.dataset_id],
      ["Selection benchmark", spec.benchmark_selection_id], ["Model version", spec.model_version_id]];
    entries.unshift([imported ? "Imported via" : "Started via", launchSourceLabel(run.job.launch_source)]);
    if (!imported && run.job.attempt > 0) entries.push(["Attempts", String(run.job.attempt)]);
    if (stats?.fit_seconds != null) entries.push(["Model fit time", duration(stats.fit_seconds)]);
    if (stats?.train_rows != null) entries.push(["Training rows", stats.train_rows.toLocaleString()]);
    if (stats?.validation_rows != null) entries.push(["Validation rows", stats.validation_rows.toLocaleString()]);
    if (run.archived_at) entries.push(["Archived", shortDate(run.archived_at)]);
    const content = JSON.stringify(entries);
    if (content !== renderedFacts) {
      $("training-facts").replaceChildren(...entries.flatMap(([key, value]) => [element("dt", key), element("dd", value || "—")]));
      renderedFacts = content;
    }
    setText($("training-settings"), run.configuration || "");
    $("training-settings").hidden = !run.configuration;
    const record = $("training-import-data");
    record.hidden = !imported;
    if (imported) {
      record.href = route(run.run_id);
      record.download = "import-" + run.run_id + ".json";
    }
    const evidence = imported ? spec : run;
    const results = evidence.external_evaluations || [], latency = evidence.external_latency;
    const currentResults = JSON.stringify([results, latency]);
    if (currentResults !== renderedEvaluations) {
      showImportedResults($("training-imported-results"), results, latency);
      $("training-external-evaluations").hidden = !results.length && !latency;
      renderedEvaluations = currentResults;
    }
    const stages = {
      starting: "Preparing training.", loading: "Loading training data.",
      "administrative-computation": "Training the model.", committing: "Saving the model and metrics.",
    };
    let status = run.job.failure?.message || "";
    if (run.job.state === "cancel_requested") status = "Cancellation requested.";
    else if (run.job.state === "queued") status = queuedStageLabel(run.job.stage, "Waiting for a worker") + ".";
    else if (run.job.state === "running") {
      status = stages[run.job.stage] || "Training the model.";
      if (run.job.attempt > 1) status += " Attempt " + run.job.attempt + " of " + run.job.max_attempts + ".";
    }
    if ($("training-stage").textContent !== status) $("training-stage").textContent = status;
    $("training-stage").hidden = !status;
    $("training-cancel").hidden = !["running", "queued"].includes(run.job.state);
    $("training-cancel").disabled = busy;
    $("training-detail").hidden = false;
  }

  async function downstream(valid) {
    const [rows, models] = await Promise.all([
      eachPage(route(run.run_id) + "/actions?include_final=true", valid),
      run.weights_deletion_id ? null : session.json(api + "models?published_only=true&limit=1"),
    ]);
    if (!valid()) return false;
    const attempts = rows.filter(row => row.job.kind === "evaluation");
    const validation = currentEvaluation(attempts.filter(row => !row.final_benchmark_id));
    const final = currentEvaluation(attempts.filter(row => row.final_benchmark_id &&
      row.final_benchmark_id === run.training.benchmark_selection_id));
    const publications = rows.filter(row => row.job.kind === "publication");
    evaluation = [final, validation].find(row => row?.job.state === "succeeded");
    showTrainingActions($("training-actions"), validation ? [validation] : [], session);
    showTrainingActions($("training-publications"), publications, session);
    $("training-downstream").hidden = false;
    const finalView = benchmark.show(run, final, valid);
    if (run.weights_deletion_id) {
      showRemovedWeights();
      await finalView;
      return false;
    }
    const newlyActive = activeModel !== undefined && activeModel !== models.active_model_version_id &&
      models.active_model_version_id === run.training.model_version_id;
    activeModel = models.active_model_version_id;
    if (newlyActive) void onPublication();
    pendingDownstream = [validation, final, ...publications].some(row => row && inProgress(row.job));
    const evaluate = $("training-evaluate");
    evaluate.hidden = Boolean(validation && (validation.job.state === "succeeded" || inProgress(validation.job)));
    evaluate.textContent = validation ? "Retry evaluation" : "Evaluate model";
    evaluate.disabled = busy || pendingDownstream;
    $("training-publish").hidden = !evaluation || activeModel === run.training.model_version_id;
    $("training-publish").disabled = busy || pendingDownstream;
    $("training-delete-weights").hidden = activeModel === run.training.model_version_id ||
      rows.some(row => inProgress(row.job) || (row.job.kind === "publication" && row.job.state === "succeeded"));
    $("training-delete-weights").disabled = busy || pendingDownstream;
    $("training-deletion-receipt").hidden = true;
    let nextStep = "";
    if (activeModel === run.training.model_version_id) nextStep = "New analyses use this model by default.";
    else if (!validation && !final) nextStep = "Evaluate the saved model before making it the default for new analyses.";
    $("training-publication").textContent = nextStep;
    $("training-publication").hidden = !nextStep;
    await finalView;
    if (valid()) $("training-final-next").disabled = busy || pendingDownstream;
    return pendingDownstream;
  }

  async function completedMetrics(valid) {
    if (metricsLoaded) return;
    const result = await session.json(route(run.run_id) + "/metrics");
    if (!valid()) return;
    const policy = isImported(run) ? run.training.configuration?.training_policy : run.training.policy;
    metrics.show(result.history, result, policy);
    $("training-metric-data").href = route(run.run_id) + "/metrics";
    $("training-metric-data").download = "training-" + run.run_id + "-metrics.json";
    $("training-metric-data").hidden = false;
    $("training-metric-error").hidden = true;
    metricsLoaded = true;
  }

  function handleRunReadError(error, retry) {
    if (!isTransientRequestError(error)) {
      $("training-sync").hidden = true;
      message(error.message, true);
      return;
    }
    $("training-sync").textContent = "Could not load this run. Retrying automatically…";
    $("training-sync").hidden = false;
    timer = setTimeout(retry, retryDelay);
    retryDelay = Math.min(retryDelay * 2, 30000);
  }

  async function refresh(stamp, fetchRun = true) {
    if (document.hidden || !selected(stamp) || !run || busy) return;
    clearTimeout(timer);
    const revision = ++pollRevision;
    const valid = () => !document.hidden && selected(stamp) && pollRevision === revision;
    const before = run.job.state;
    try {
      if (fetchRun) {
        const next = await readRun(run.run_id);
        if (!valid()) return;
        if (next.name_revision > run.name_revision) applyName(run.run_id, next.display_name, next.name_revision);
        run = withSavedChanges(next, run);
      }
      facts();
      if (run.job.state !== before) void list();
      let waiting = inProgress(run.job);
      if (["running", "cancel_requested"].includes(run.job.state)) {
        // A slow history page must not delay cancellation or completion updates.
        void progress.poll(route(run.run_id) + "/progress", run.job.attempt, run.training.policy);
      } else {
        progress.stop();
        if (run.job.state !== "succeeded") metrics.clear();
      }
      if (run.job.state === "succeeded") {
        const [charts, actions] = await Promise.allSettled([completedMetrics(valid), isImported(run) ? false : downstream(valid)]);
        if (!valid()) return;
        if (actions.status === "rejected") throw actions.reason;
        waiting = actions.value;
        if (charts.status === "rejected") {
          const error = charts.reason;
          if (isTransientRequestError(error)) throw error;
          setText($("training-metric-error"), "Could not load training metrics. " + error.message);
          $("training-metric-error").hidden = false;
        }
      }
      if (!valid()) return;
      $("training-sync").hidden = true;
      retryDelay = 2500;
      if (waiting) timer = setTimeout(() => refresh(stamp), 2500);
    } catch (error) {
      if (valid()) handleRunReadError(error, () => refresh(stamp));
    }
  }

  async function show(id = null, filter = "active", ids = []) {
    if (!isAdmin()) return;
    if (id === "new" && acceptedDraftRun) {
      const accepted = acceptedDraftRun;
      acceptedDraftRun = null;
      return navigate(runPath(accepted, filter), true);
    }
    const entering = !opened;
    const viewChanged = filter !== view;
    const pairChanged = id === "compare" && ids.join() !== comparedIds.join();
    comparedIds = id === "compare" ? ids : [];
    const filterFocused = document.activeElement === $("training-history-view");
    if (entering) {
      opened = true;
      epoch++;
      $("training").hidden = false;
    } else if (id === target && !viewChanged && !pairChanged) return;
    if (entering || viewChanged) {
      view = filter;
      $("training-history-view").value = view;
      offset = 0;
      archive.reset();
      if (entering) $("training-empty").hidden = $("training-prev").hidden = $("training-next").hidden = true;
    }
    if (!entering && id === target && !pairChanged) {
      void list();
      return;
    }
    if (!entering) deletion.close();
    target = id;
    runsToggle.setAttribute("aria-expanded", String(id === null));
    deselect();
    lock(false);
    message();
    if (entering || viewChanged || id === null) void list(entering && id === null);
    if (id === null) {
      if (!filterFocused) $("training-title").focus();
      return;
    }
    if (id === "new") {
      void form.open();
      $("training-config-title").focus();
      return;
    }
    if (id === "compare") return comparison.show(comparedIds);
    markCurrent(id);
    return loadRun(id, selection, listed.get(id), !filterFocused);
  }

  async function loadRun(id, stamp, cached, focus = false) {
    if (document.hidden || !selected(stamp)) return;
    clearTimeout(timer);
    try {
      const row = cached || await readRun(id);
      if (!selected(stamp)) return;
      $("training-sync").hidden = true;
      retryDelay = 2500;
      run = withSavedChanges(row, listed.get(id));
      const recheckRun = Boolean(cached && finished(row.job));
      if (recheckRun) facts();
      void refresh(stamp, recheckRun);
      if (focus && document.activeElement !== $("training-history-view")) $("training-detail-title").focus();
    } catch (error) {
      if (selected(stamp)) handleRunReadError(error, () => loadRun(id, stamp));
    }
  }

  function lock(locked) {
    busy = locked;
    if (locked) {
      clearTimeout(timer);
      pollRevision++;
    }
    for (const id of ["training-new", "training-refresh", "training-cancel", "training-copy-run"])
      $(id).disabled = locked;
    $("training-evaluate").disabled = $("training-publish").disabled =
      $("training-final-next").disabled = $("training-delete-weights").disabled = locked || pendingDownstream;
    form.setBusy(locked || submittingDraft === draftRevision);
  }

  async function action(work) {
    if (!opened || busy) return;
    const stamp = selection;
    lock(true);
    message();
    try { await work(stamp); }
    catch (error) { if (selected(stamp)) message(error.message, true); }
    finally { if (selected(stamp)) { lock(false); if (run) void refresh(selection); } }
  }

  function chooseFromHistory(id) {
    if (id !== target) return navigate(runPath(id));
    runsToggle.setAttribute("aria-expanded", "false");
    $(id === "new" ? "training-config-title" : "training-detail-title").focus();
  }
  runsToggle.addEventListener("click", () => {
    runsToggle.setAttribute("aria-expanded", String(runsToggle.getAttribute("aria-expanded") !== "true"));
  });
  $("training-history-view").addEventListener("change", event => {
    void navigate(runPath(target, event.target.value));
  });
  function discardDraft() {
    draftRequests = new Map();
    draftRevision++;
    editedDraft = false;
    form.discard();
  }
  $("training-new").addEventListener("click", () => {
    if (target !== "new" && submittingDraft === draftRevision) discardDraft();
    acceptedDraftRun = null;
    chooseFromHistory("new");
  });
  $("training-copy-run").addEventListener("click", () => {
    if (!run || busy || isImported(run)) return;
    if (editedDraft && !window.confirm("Replace your unfinished training draft with settings from this run?")) return;
    discardDraft();
    acceptedDraftRun = null;
    form.useSavedRun(run.configuration);
    chooseFromHistory("new");
  });
  function draftEdited() {
    draftRevision++;
    editedDraft = true;
    form.setBusy(busy);
  }
  $("training-form").addEventListener("input", draftEdited);
  $("training-form").addEventListener("change", draftEdited);
  $("training-form").addEventListener("submit", event => {
    event.preventDefault();
    if ($("training-submit").disabled) return;
    void action(async () => {
      const body = form.payload(), revision = draftRevision;
      submittingDraft = revision;
      let record;
      try {
        record = await post(api + "training-runs", body, draftRequests);
      } finally {
        if (submittingDraft === revision) {
          submittingDraft = null;
          form.setBusy(busy);
        }
      }
      if (revision !== draftRevision) {
        if (opened) void list();
        return;
      }
      discardDraft();
      if (!opened || target !== "new") {
        // Back to the submitted form should open the accepted run. A new draft starts separately.
        acceptedDraftRun = record.run_id;
        if (opened) void list();
        return;
      }
      offset = 0;
      listed.set(record.run_id, record);
      navigate(runPath(record.run_id, "active"), true);
      void list();
    });
  });
  $("training-cancel").addEventListener("click", () => action(async stamp => {
    const job = await session.json(api + "jobs/" + encodeURIComponent(run.run_id) + "/cancel", {method: "POST"});
    if (selected(stamp)) {
      run = {...run, job};
      facts();
      void list();
    }
  }));
  $("training-evaluate").addEventListener("click", () => action(async stamp => {
    if (run.weights_deletion_id) return;
    const spec = run.training;
    await post(api + "evaluations", {model_version_id: spec.model_version_id, dataset_id: spec.dataset_id,
      reference_set_id: spec.validation_reference_set_id, partition: "validation"});
    if (selected(stamp)) pendingDownstream = true;
  }));
  $("training-publish").addEventListener("click", () => action(async (stamp) => {
    const id = run.training.model_version_id;
    if (run.weights_deletion_id || !evaluation || activeModel === undefined) return;
    const confirmation = "Use " + id + " as the default for new analyses?" +
      (activeModel ? "\nThis replaces " + activeModel + "." : "");
    if (!window.confirm(confirmation)) return;
    await post(api + "models/" + encodeURIComponent(id) + "/publish",
      {evaluation_run_id: evaluation.run_id, expected_previous: activeModel});
    if (selected(stamp)) {
      pendingDownstream = true;
      message("Publication queued.");
    }
  }));
  $("training-delete-weights").addEventListener("click", () => {
    if (run && !run.weights_deletion_id)
      void deletion.open("model_weights", run.run_id, run.display_name || run.training.model_version_id,
        {archived: Boolean(run.archived_at)});
  });
  $("training-deletion-receipt").addEventListener("click", () => {
    if (run?.weights_deletion_id)
      void deletion.showReceipt("model_weights", run.run_id, run.weights_deletion_id, run.display_name || run.training.model_version_id);
  });
  for (const [id, delta] of [["training-prev", -pageSize], ["training-next", pageSize], ["training-refresh", 0]])
    $(id).addEventListener("click", () => {
      offset = Math.max(0, offset + delta);
      message();
      void list();
      if (!delta) void refreshDetail();
    });

  function refreshAfterReturn() {
    if (!opened) return;
    void list();
    if (target && target !== "new" && target !== "compare") void refreshDetail();
  }

  return {close, reset, show, weightsRemoved, refreshAfterReturn};
}
