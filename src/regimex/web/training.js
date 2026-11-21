import {$, analysisStateLabel} from "./ui.js";
import {showTrainingMetrics} from "./training-metrics.js";
import {architectureName, createTrainingForm} from "./training-form.js";

const api = "/api/v1/";
const pageSize = 12;
const inProgress = (job) => ["queued", "running", "cancel_requested"].includes(job.state);
const route = (id) => api + "training-runs/" + encodeURIComponent(id);

function text(tag, value) {
  const el = document.createElement(tag);
  el.textContent = value;
  return el;
}

export function createTraining({session, navigate, onPublication}) {
  let opened = false, epoch = 0, selection = 0, listRequest = 0;
  let offset = 0, timer = null, busy = false, pollRevision = 0;
  let run = null, evaluation = null, activeModel, metricsLoaded = false;
  let pendingDownstream = false, retryDelay = 2500, target = null;
  let listed = new Map();
  const form = createTrainingForm({session, eachPage});
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

  async function list(preselect = false) {
    const stamp = epoch, request = ++listRequest;
    const page = await session.json(api + "training-runs?limit=" + (pageSize + 1) + "&offset=" + offset);
    if (!current(stamp) || request !== listRequest) return;
    const focused = document.activeElement?.dataset?.runId;
    const rows = page.items.slice(0, pageSize);
    listed = new Map(rows.map(row => [row.run_id, row]));
    $("training-runs").replaceChildren(...rows.map((row) => {
      const item = document.createElement("li");
      const button = text("button", "");
      button.type = "button";
      button.className = "quiet";
      button.dataset.runId = row.run_id;
      button.setAttribute("aria-current", String(row.run_id === target));
      const time = new Date(row.job.created_at).toLocaleString();
      button.append(text("strong", architectureName(row.training.architecture_id, true) + " · " + analysisStateLabel(row.job.state)),
        text("span", time), text("small", row.run_id.slice(0, 8)));
      button.addEventListener("click", () => chooseFromHistory(row.run_id));
      item.append(button);
      return item;
    }));
    if (focused) {
      for (const li of $("training-runs").children)
        if (li.children[0].dataset.runId === focused) li.children[0].focus({preventScroll: true});
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

  function facts() {
    const spec = run.training;
    $("training-detail-title").textContent = architectureName(spec.architecture_id, true) + " training";
    $("training-state").textContent = analysisStateLabel(run.job.state);
    $("training-state").dataset.state = run.job.state;
    $("training-created").textContent = "Created " + new Date(run.job.created_at).toLocaleString();
    const entries = [["Run", run.run_id], ["Dataset", spec.dataset_id], ["Selection benchmark", spec.benchmark_selection_id],
      ["Model version", spec.model_version_id]];
    $("training-facts").replaceChildren(...entries.flatMap(([key, value]) => [text("dt", key), text("dd", value || "—")]));
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
    $("training-actions").replaceChildren(...rows.map(row => {
      const item = text("li", (row.job.kind === "evaluation" ? "Validation" : "Publication") + " · " +
        analysisStateLabel(row.job.state) + (row.job.failure ? " · " + row.job.failure.message : ""));
      if (row.job.kind === "evaluation" && row.job.state === "succeeded") {
        const link = text("a", "Evaluation JSON");
        link.href = api + "evaluations/" + encodeURIComponent(row.run_id) + "/result";
        link.download = "evaluation-" + row.run_id + ".json";
        item.append(document.createTextNode(" · "), link);
      }
      return item;
    }));
    pendingDownstream = rows.some(row => inProgress(row.job));
    $("training-evaluate").disabled = busy || pendingDownstream;
    $("training-publish").hidden = !evaluation || activeModel === run.training.model_version_id;
    $("training-publish").disabled = busy || pendingDownstream;
    $("training-publication").textContent = activeModel === run.training.model_version_id ?
      "This candidate is active." : evaluation ? "Current active model: " + (activeModel || "none") :
      "Evaluate this fixed candidate on its validation references before publishing.";
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
        run = next;
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
      run = row;
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
    if (!evaluation || activeModel === undefined || !window.confirm("Activate " + id + "? Current model: " + (activeModel || "none"))) return;
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
