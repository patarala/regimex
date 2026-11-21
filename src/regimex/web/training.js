import {$, analysisStateLabel, element, shortDate} from "./ui.js";
import {showTrainingActions} from "./training-evaluation.js";
import {showTrainingMetrics} from "./training-metrics.js";
import {architectureName, createTrainingForm, datasetName} from "./training-form.js";
import {createRunNameEditor} from "./training-name.js";

const api = "/api/v1/";
const pageSize = 12;
const inProgress = (job) => ["queued", "running", "cancel_requested"].includes(job.state);
const route = (id) => api + "training-runs/" + encodeURIComponent(id);

function runContext(row) {
  return architectureName(row.training.architecture_id, true) + " · " + datasetName(row.training.dataset_id);
}

// A polling response can have been read before a rename in this or another tab.
function withLatestName(row, known) {
  if ((known?.name_revision ?? 0) <= (row.name_revision ?? 0)) return row;
  return {...row, display_name: known.display_name, name_revision: known.name_revision};
}

export function createTraining({session, navigate, onPublication}) {
  let opened = false, epoch = 0, selection = 0, listRequest = 0;
  let offset = 0, timer = null, busy = false, pollRevision = 0;
  let run = null, evaluation = null, activeModel, metricsLoaded = false;
  let pendingDownstream = false, retryDelay = 2500, target = null;
  let listed = new Map();
  const form = createTrainingForm({session, eachPage});
  const names = createRunNameEditor({session, getRun: () => run, applyName});
  const runsToggle = $("training-runs-toggle");
  const runHash = id => "#training/" + encodeURIComponent(id);
  const pending = new Map();
  const isAdmin = () => session.value?.principal.role === "admin";
  const current = (stamp) => opened && stamp === epoch && isAdmin();
  const selected = (stamp) => opened && stamp === selection && isAdmin();

  function message(value = "", error = false) {
    $("training-notice").textContent = value;
    $("training-notice").classList.toggle("error", error);
    $("training-notice").hidden = !value;
  }

  function deselect() {
    selection++;
    clearTimeout(timer);
    form.close();
    names.close();
    retryDelay = 2500;
    $("training-sync").hidden = true;
    $("training-run-details").open = false;
    for (const li of $("training-runs").children) li.children[0].setAttribute("aria-current", "false");
    run = evaluation = null;
    activeModel = undefined;
    pendingDownstream = false;
    metricsLoaded = false;
    $("training-metrics").hidden = $("training-downstream").hidden = $("training-detail").hidden = true;
    for (const id of ["training-state", "training-stage", "training-settings", "training-score",
      "training-publication"]) $(id).textContent = "";
    $("training-detail-title").textContent = "Run";
    $("training-metric-data").removeAttribute("href");
    $("training-metric-data").removeAttribute("download");
    $("training-facts").replaceChildren();
    $("training-charts").replaceChildren();
    $("training-seed").replaceChildren();
    $("training-seed").onchange = null;
    $("training-actions").replaceChildren();
  }

  function close() {
    epoch++;
    opened = false;
    target = null;
    listed.clear();
    busy = false;
    deselect();
    $("training").hidden = true;
    $("training-config").hidden = true;
    $("training-runs").replaceChildren();
    message();
  }

  function reset() {
    close();
    form.discard();
    pending.clear();
    $("open-training").hidden = true;
  }

  function reportIfCurrent(stamp, error) {
    if (current(stamp)) message(error.message, true);
  }

  function renderRows() {
    const focused = document.activeElement?.dataset?.runId;
    const rows = [...listed.values()].map(row => ({
      row, context: runContext(row), date: shortDate(row.job.created_at),
    }));
    $("training-runs").replaceChildren(...rows.map(({row, context, date}) => {
      const item = document.createElement("li");
      const button = element("button", "");
      button.type = "button";
      button.className = "quiet";
      button.dataset.runId = row.run_id;
      button.setAttribute("aria-current", String(row.run_id === target));
      const time = element("time", date);
      time.dateTime = row.job.created_at;
      const sameLabel = rows.some(other => other.row !== row && other.row.display_name === row.display_name &&
        other.context === context && other.date === date);
      if (sameLabel)
        time.append(" · …" + row.run_id.slice(-8));
      const title = element("strong", row.display_name || context);
      title.dir = "auto";
      button.title = title.textContent;
      button.append(title);
      if (row.display_name) button.append(element("span", context, "training-run-context"));
      const state = element("span", analysisStateLabel(row.job.state));
      state.dataset.state = row.job.state;
      button.append(time, state);
      button.addEventListener("click", () => chooseFromHistory(row.run_id));
      item.append(button);
      return item;
    }));
    if (focused) {
      for (const li of $("training-runs").children)
        if (li.children[0].dataset.runId === focused) li.children[0].focus({preventScroll: true});
    }
  }

  async function list(preselect = false) {
    const stamp = epoch, request = ++listRequest;
    const page = await session.json(api + "training-runs?limit=" + (pageSize + 1) + "&offset=" + offset);
    if (!current(stamp) || request !== listRequest) return;
    const rows = page.items.slice(0, pageSize).map(row =>
      withLatestName(row, run?.run_id === row.run_id ? run : listed.get(row.run_id)));
    listed = new Map(rows.map(row => [row.run_id, row]));
    renderRows();
    if (run && listed.has(run.run_id)) {
      run = withLatestName(run, listed.get(run.run_id));
      showTitle();
    }
    $("training-empty").hidden = page.items.length !== 0;
    $("training-prev").hidden = offset === 0;
    $("training-next").hidden = page.items.length <= pageSize;
    if (preselect && target === null && rows.length) navigate(runHash(rows[0].run_id), true);
  }

  // A network error can arrive after the server committed. Reuse that exact request's key.
  async function post(path, body) {
    const request = JSON.stringify([path, body]);
    const key = pending.get(request) || crypto.randomUUID();
    pending.set(request, key);
    const result = await session.json(path, {method: "POST", body, headers: {"Idempotency-Key": key}});
    if (pending.get(request) === key) pending.delete(request);
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
    $("training-detail-title").textContent = run.display_name || context;
    $("training-created").textContent = (run.display_name ? context + " · " : "") + "Created " + shortDate(run.job.created_at);
  }

  function applyName(id, name, revision) {
    const saved = {display_name: name, name_revision: revision};
    if (run?.run_id === id) {
      run = withLatestName(run, saved);
      showTitle();
    }
    if (listed.has(id)) {
      listed.set(id, withLatestName(listed.get(id), saved));
      renderRows();
    }
  }

  function facts() {
    const spec = run.training;
    showTitle();
    $("training-state").textContent = analysisStateLabel(run.job.state);
    $("training-state").dataset.state = run.job.state;
    const entries = [["Run", run.run_id], ["Dataset", spec.dataset_id], ["Selection benchmark", spec.benchmark_selection_id],
      ["Model version", spec.model_version_id]];
    $("training-facts").replaceChildren(...entries.flatMap(([key, value]) => [element("dt", key), element("dd", value || "—")]));
    $("training-settings").textContent = JSON.stringify({parameters: spec.parameters, policy: spec.policy,
      training_runtime: spec.training_runtime, inference_runtime: spec.inference_runtime,
      analysis_profile_id: spec.analysis_profile_id, labels_reference_set_id: spec.labels_reference_set_id,
      validation_reference_set_id: spec.validation_reference_set_id}, null, 2);
    const stages = {
      queued: "Waiting for a worker.", loading: "Loading training data.",
      "administrative-computation": "Training the model.", committing: "Saving the model and metrics.",
      retry_wait: "Waiting to retry.",
    };
    let status = run.job.failure?.message || "";
    if (run.job.state === "cancel_requested") status = "Cancellation requested.";
    else if (inProgress(run.job)) {
      status = stages[run.job.stage] || (run.job.state === "queued" ? stages.queued : "Training the model.");
      if (run.job.attempt > 1) status += " Attempt " + run.job.attempt + " of " + run.job.max_attempts + ".";
      status += " Charts appear when training finishes.";
    }
    if ($("training-stage").textContent !== status) $("training-stage").textContent = status;
    $("training-stage").hidden = !status;
    $("training-cancel").hidden = !["running", "queued"].includes(run.job.state);
    $("training-cancel").disabled = busy;
    $("training-detail").hidden = false;
  }

  async function downstream(valid) {
    const [rows, models] = await Promise.all([
      eachPage(route(run.run_id) + "/actions", valid),
      session.json(api + "models?published_only=true&limit=1"),
    ]);
    if (!valid()) return false;
    evaluation = rows.find(row => row.job.kind === "evaluation" && row.job.state === "succeeded");
    const newlyActive = activeModel !== undefined && activeModel !== models.active_model_version_id &&
      models.active_model_version_id === run.training.model_version_id;
    activeModel = models.active_model_version_id;
    if (newlyActive) void onPublication();
    showTrainingActions($("training-actions"), rows, session);
    const outstanding = rows.find(row => inProgress(row.job));
    pendingDownstream = Boolean(outstanding);
    $("training-evaluate").disabled = busy || pendingDownstream;
    $("training-publish").hidden = !evaluation || activeModel === run.training.model_version_id;
    $("training-publish").disabled = busy || pendingDownstream;
    let nextStep = "Evaluate the saved model on its validation data before making it the default for new analyses.";
    if (outstanding) {
      nextStep = outstanding.job.kind === "evaluation" ? "Evaluating the saved model." : "Updating the default model.";
    } else if (activeModel === run.training.model_version_id) {
      nextStep = "New analyses use this model by default.";
    } else if (evaluation) {
      nextStep = "Evaluation complete. You can now make this model the default for new analyses.";
    }
    $("training-publication").textContent = nextStep;
    $("training-downstream").hidden = false;
    return pendingDownstream;
  }

  async function refresh(stamp, fetchRun = true) {
    if (!selected(stamp) || !run || busy) return;
    clearTimeout(timer);
    const revision = ++pollRevision, workspace = epoch;
    const valid = () => selected(stamp) && pollRevision === revision;
    const before = run.job.state;
    try {
      if (fetchRun) {
        const next = await session.json(route(run.run_id));
        if (!valid()) return;
        if (next.name_revision > run.name_revision) applyName(run.run_id, next.display_name, next.name_revision);
        run = withLatestName(next, run);
      }
      facts();
      if (run.job.state !== before) void list().catch(error => reportIfCurrent(workspace, error));
      let waiting = inProgress(run.job);
      if (run.job.state === "succeeded") {
        if (!metricsLoaded) {
          const metrics = await session.json(route(run.run_id) + "/metrics");
          if (!valid()) return;
          showTrainingMetrics(metrics);
          $("training-metric-data").href = route(run.run_id) + "/metrics";
          $("training-metric-data").download = "training-" + run.run_id + "-metrics.json";
          metricsLoaded = true;
        }
        waiting = await downstream(valid);
      }
      if (!valid()) return;
      $("training-sync").hidden = true;
      retryDelay = 2500;
      if (waiting) timer = setTimeout(() => refresh(stamp), 2500);
    } catch (error) {
      if (!valid()) return;
      if (error.status && error.status !== 429 && error.status < 500) {
        $("training-sync").hidden = true;
        message(error.message, true);
        return;
      }
      $("training-sync").textContent = "Could not update this run. Retrying automatically…";
      $("training-sync").hidden = false;
      timer = setTimeout(() => refresh(stamp), retryDelay);
      retryDelay = Math.min(retryDelay * 2, 30000);
    }
  }

  async function show(id = null) {
    if (!isAdmin()) return;
    const entering = !opened;
    if (entering) {
      opened = true;
      epoch++;
      offset = 0;
      $("training").hidden = false;
    } else if (id === target) return;
    target = id;
    runsToggle.setAttribute("aria-expanded", String(id === null));
    deselect();
    lock(false);
    message();
    const stamp = selection, workspace = epoch;
    if (entering || id === null) void list(id === null).catch(error => reportIfCurrent(workspace, error));
    if (id === null) {
      $("training-title").focus();
      return;
    }
    if (id === "new") {
      void form.open();
      $("training-config-title").focus();
      return;
    }
    for (const li of $("training-runs").children) {
      const button = li.children[0];
      button.setAttribute("aria-current", String(button.dataset.runId === id));
    }
    try {
      const row = listed.get(id) || await session.json(route(id));
      if (!selected(stamp)) return;
      run = withLatestName(row, listed.get(id));
      void refresh(stamp, false);
      $("training-detail-title").focus();
    } catch (error) {
      if (selected(stamp)) message(error.message, true);
    }
  }

  function lock(locked) {
    busy = locked;
    if (locked) {
      clearTimeout(timer);
      pollRevision++;
    }
    for (const id of ["training-new", "training-refresh", "training-cancel"])
      $(id).disabled = locked;
    $("training-evaluate").disabled = $("training-publish").disabled = locked || pendingDownstream;
    form.setBusy(locked);
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
    if (id !== target) return navigate(runHash(id));
    runsToggle.setAttribute("aria-expanded", "false");
    $(id === "new" ? "training-config-title" : "training-detail-title").focus();
  }
  runsToggle.addEventListener("click", () => {
    runsToggle.setAttribute("aria-expanded", String(runsToggle.getAttribute("aria-expanded") !== "true"));
  });
  $("training-new").addEventListener("click", () => chooseFromHistory("new"));
  $("training-form").addEventListener("submit", event => {
    event.preventDefault();
    if ($("training-submit").disabled) return;
    void action(async stamp => {
      const record = await post(api + "training-runs", form.payload());
      if (!selected(stamp)) return;
      offset = 0;
      form.discard();
      listed.set(record.run_id, record);
      navigate(runHash(record.run_id), true);
      const workspace = epoch;
      void list().catch(error => reportIfCurrent(workspace, error));
    });
  });
  $("training-cancel").addEventListener("click", () => action(async stamp => {
    const job = await session.json(api + "jobs/" + encodeURIComponent(run.run_id) + "/cancel", {method: "POST"});
    if (selected(stamp)) {
      run = {...run, job};
      facts();
      const workspace = epoch;
      void list().catch(error => reportIfCurrent(workspace, error));
    }
  }));
  $("training-evaluate").addEventListener("click", () => action(async stamp => {
    const spec = run.training;
    await post(api + "evaluations", {model_version_id: spec.model_version_id, dataset_id: spec.dataset_id,
      reference_set_id: spec.validation_reference_set_id, partition: "validation"});
    if (selected(stamp)) pendingDownstream = true;
  }));
  $("training-publish").addEventListener("click", () => action(async (stamp) => {
    const id = run.training.model_version_id;
    if (!evaluation || activeModel === undefined) return;
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
  for (const [id, delta] of [["training-prev", -pageSize], ["training-next", pageSize], ["training-refresh", 0]])
    $(id).addEventListener("click", () => {
      offset = Math.max(0, offset + delta);
      const stamp = epoch;
      message();
      void list(target === null).catch(error => reportIfCurrent(stamp, error));
      if (!delta && run) void refresh(selection);
      else if (!delta && target && target !== "new") {
        const id = target;
        target = null;
        void show(id);
      }
    });

  return {close, reset, show};
}
