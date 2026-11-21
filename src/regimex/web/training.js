import {$, analysisStateLabel, option} from "./ui.js";
import {showTrainingMetrics} from "./training-metrics.js";

const api = "/api/v1/";
const pageSize = 12;
const inProgress = (job) => ["queued", "running", "cancel_requested"].includes(job.state);
const route = (id) => api + "training-runs/" + encodeURIComponent(id);

function text(tag, value) {
  const el = document.createElement(tag);
  el.textContent = value;
  return el;
}

function options(id, items, empty) {
  $(id).replaceChildren(...(items.length ? items.map(([value, label]) => option(value, label)) : [option("", empty)]));
}

function readSettings(id, required = true) {
  const field = $(id), input = field.value.trim();
  if (!input && !required) return undefined;
  try {
    const value = JSON.parse(input);
    if (value && typeof value === "object" && !Array.isArray(value)) return value;
  } catch {
    // Identify the editable field instead of exposing a parser-specific error.
  }
  field.closest("details").open = true;
  field.focus();
  throw new Error("Enter a JSON object for " + field.labels[0].textContent.trim() + ".");
}

export function createTraining({session, onPublication}) {
  let opened = false, epoch = 0, selection = 0, listRequest = 0, referenceRequest = 0;
  let offset = 0, timer = null, busy = false, pollRevision = 0;
  let run = null, evaluation = null, activeModel, metricsLoaded = false;
  let architectures = [], pendingDownstream = false;
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
    busy = false;
    deselect();
    $("training").hidden = true;
    $("training-config").hidden = true;
    $("training-runs").replaceChildren();
    message();
  }

  function reset() {
    close();
    pending.clear();
    architectures = [];
    $("training-benchmark").value = "";
    $("training-parameters").value = "{}";
    $("training-policy").value = "";
    $("training-runtime").value = $("training-inference").value = "";
    for (const id of ["training-dataset", "training-architecture", "training-labels", "training-validation"])
      $(id).replaceChildren();
    $("open-training").hidden = true;
  }

  function reportIfCurrent(stamp, error) {
    if (current(stamp)) message(error.message, true);
  }

  async function list(preselect = false) {
    const stamp = epoch, request = ++listRequest;
    const page = await session.json(api + "training-runs?limit=" + (pageSize + 1) + "&offset=" + offset);
    if (!current(stamp) || request !== listRequest) return;
    $("training-runs").replaceChildren(...page.items.slice(0, pageSize).map((row) => {
      const item = document.createElement("li");
      const button = text("button", "");
      button.type = "button";
      button.className = "quiet";
      button.dataset.runId = row.run_id;
      button.setAttribute("aria-current", String(row.run_id === run?.run_id));
      const time = new Date(row.job.created_at).toLocaleString();
      button.append(text("strong", row.training.architecture_id + " · " + analysisStateLabel(row.job.state)),
        text("span", time), text("small", row.run_id.slice(0, 8)));
      button.addEventListener("click", () => select(row));
      item.append(button);
      return item;
    }));
    $("training-empty").hidden = page.items.length !== 0;
    $("training-prev").hidden = offset === 0;
    $("training-next").hidden = page.items.length <= pageSize;
    if (preselect && !run && $("training-config").hidden && page.items.length) select(page.items[0]);
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

  function submitReady() {
    $("training-submit").disabled = busy || ![$("training-dataset"), $("training-architecture"),
      $("training-labels"), $("training-validation"), $("training-benchmark")].every(el => el.value.trim());
  }

  async function references(stamp) {
    const request = ++referenceRequest;
    const valid = () => selected(stamp) && request === referenceRequest;
    options("training-labels", [], "No training labels");
    options("training-validation", [], "No validation references");
    submitReady();
    if (!$("training-dataset").value) return;
    const rows = await eachPage(api + "reference-sets?dataset_id=" + encodeURIComponent($("training-dataset").value), valid);
    if (!valid()) return;
    for (const [id, partition] of [["training-labels", "train"], ["training-validation", "validation"]]) {
      options(id, rows.filter(row => row.partition === partition && row.current_training_profile)
        .map(row => [row.reference_set_id, row.kind + " · " + row.reference_set_id]), "No " + partition + " references");
    }
    submitReady();
  }

  function policy() {
    const arch = architectures.find(a => a.architecture_id === $("training-architecture").value);
    $("training-policy").value = arch ? JSON.stringify(arch.default_training_policy, null, 2) : "";
    $("training-parameters").value = "{}";
    submitReady();
  }

  async function configure() {
    deselect();
    const stamp = selection;
    message();
    $("training-config").hidden = false;
    options("training-dataset", [], "Loading datasets…");
    options("training-architecture", [], "Loading architectures…");
    submitReady();
    const valid = () => selected(stamp);
    try {
      const [datasets, backends] = await Promise.all([eachPage(api + "datasets", valid), eachPage(api + "architectures", valid)]);
      if (!valid()) return;
      architectures = backends.filter(row => row.executable !== false);
      const eligibleDatasets = datasets.filter(row => row.partitions.train && row.partitions.validation);
      options("training-dataset", eligibleDatasets.map(row => [row.dataset_id, row.dataset_id + " · " +
        row.partitions.train + " train / " + row.partitions.validation + " validation"]), "No development datasets");
      options("training-architecture", architectures.map(row => [row.architecture_id, row.architecture_id]),
        "No training backends");
      policy();
      await references(stamp);
    } catch (error) { if (valid()) message(error.message, true); }
  }

  function facts() {
    const spec = run.training;
    $("training-detail-title").textContent = spec.architecture_id + " training";
    $("training-state").textContent = analysisStateLabel(run.job.state);
    const entries = [["Run", run.run_id], ["Dataset", spec.dataset_id], ["Selection benchmark", spec.benchmark_selection_id],
      ["Model version", spec.model_version_id], ["Created", new Date(run.job.created_at).toLocaleString()]];
    $("training-facts").replaceChildren(...entries.flatMap(([key, value]) => [text("dt", key), text("dd", value || "—")]));
    $("training-settings").textContent = JSON.stringify({parameters: spec.parameters, policy: spec.policy,
      training_runtime: spec.training_runtime, inference_runtime: spec.inference_runtime,
      analysis_profile_id: spec.analysis_profile_id, labels_reference_set_id: spec.labels_reference_set_id,
      validation_reference_set_id: spec.validation_reference_set_id}, null, 2);
    $("training-stage").textContent = run.job.failure?.message ||
      (inProgress(run.job) ? "Stage: " + run.job.stage + " · Attempt " + run.job.attempt + " of " + run.job.max_attempts : "");
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
      if (valid() && waiting) timer = setTimeout(() => refresh(stamp), 2500);
    } catch (error) { if (valid()) message(error.message, true); }
  }

  function select(row) {
    deselect();
    message();
    $("training-config").hidden = true;
    run = row;
    for (const li of $("training-runs").children) {
      const button = li.children[0];
      button.setAttribute("aria-current", String(button.dataset.runId === row.run_id));
    }
    void refresh(selection, false);
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
    submitReady();
  }

  async function action(work) {
    if (!opened || busy) return;
    const stamp = selection, workspace = epoch;
    lock(true);
    message();
    try { await work(stamp); }
    catch (error) { if (selected(stamp)) message(error.message, true); }
    finally { if (current(workspace)) { lock(false); if (run) void refresh(selection); } }
  }

  $("open-training").addEventListener("click", () => {
    if (!isAdmin()) return;
    opened = true;
    epoch++;
    offset = 0;
    lock(false);
    $("workspace").hidden = $("open-connections").hidden = $("open-training").hidden = true;
    $("training").hidden = false;
    const stamp = epoch;
    void list(true).catch(error => reportIfCurrent(stamp, error));
  });
  $("close-training").addEventListener("click", () => {
    close();
    $("workspace").hidden = !session.value;
    $("open-connections").hidden = $("open-training").hidden = !isAdmin();
  });
  $("training-new").addEventListener("click", configure);
  $("training-dataset").addEventListener("change", () => {
    const stamp = selection;
    void references(stamp).catch(error => { if (selected(stamp)) message(error.message, true); });
  });
  $("training-architecture").addEventListener("change", policy);
  $("training-benchmark").addEventListener("input", submitReady);
  $("training-form").addEventListener("submit", (event) => {
    event.preventDefault();
    void action(async stamp => {
      const body = {
        dataset_id: $("training-dataset").value, architecture_id: $("training-architecture").value,
        labels_reference_set_id: $("training-labels").value, validation_reference_set_id: $("training-validation").value,
        benchmark_selection_id: $("training-benchmark").value,
        policy: readSettings("training-policy"), parameters: readSettings("training-parameters"),
        training_runtime: readSettings("training-runtime", false),
        inference_runtime: readSettings("training-inference", false),
      };
      const record = await post(api + "training-runs", body);
      if (!selected(stamp)) return;
      offset = 0;
      select(record);
      await list();
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
      void list(!run && $("training-config").hidden).catch(error => reportIfCurrent(stamp, error));
      if (!delta && run) void refresh(selection);
    });

  return {activate() { reset(); $("open-training").hidden = !isAdmin(); }, reset};
}
