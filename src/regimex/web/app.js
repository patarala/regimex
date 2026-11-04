"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const MAX_BYTES = 16 * 1024 * 1024;
  const HISTORY_PAGE_SIZE = 20;
  const MODEL_PAGE_SIZE = 100;
  const COMPARISON_PAGE_SIZE = 20;
  const QUEUE_CHECK_INTERVAL = 5000;
  let session = null;
  let exportText = null;
  let snapshot = null;
  let current = null;
  let result = null;
  let timer = null;
  let epoch = 0;
  let retryAction = null;
  let busy = false;
  let fileEpoch = 0;
  let historyGeneration = 0;
  let historyOffset = 0;
  let historyHasMore = false;
  let historyQuery = "";
  let historyState = "";
  let historyItems = [];
  let historyPendingOffset = null;
  let models = [];
  let activeModelId = null;
  let modelOffset = 0;
  let modelHasMore = false;
  let modelGeneration = 0;
  let modelsReady = false;
  let modelsLoading = false;
  let comparisonGeneration = 0;
  let comparisonOffset = 0;
  let comparisonHasMore = false;
  let comparisonItems = [];
  let queueStatus = null;
  let queueCheckedAt = -Infinity;
  let queueCheckPending = null;
  const stateLabels = {
    queued: "Queued", running: "Running", cancel_requested: "Cancelling",
    cancelled: "Cancelled", succeeded: "Complete", failed: "Failed",
  };
  const stages = {
    queued: "Waiting to start.",
    loading: "Loading the experiment.",
    predicting: "Analyzing prices.",
    rendering: "Preparing the report.",
    committing: "Saving the analysis.",
    retry_wait: "Waiting to retry.",
  };
  const failureActions = {
    invalid_data: "Check the export and upload a corrected file.",
    unsupported_configuration: "Ask your administrator to check the model and worker configuration.",
    incompatible_artifact: "Ask your administrator to verify or replace the saved model and data.",
    capacity_limit: "Ask your administrator to check worker and storage capacity before running again.",
    source_failure: "Check that the source is available before starting a new analysis.",
    infrastructure_failure: "Try a new analysis once the service is available. Contact your administrator if this continues.",
    attempts_exhausted: "Ask your administrator to check worker availability before starting a new analysis.",
  };
  function requestError(data) {
    const error = data?.error;
    const message = error?.message || "The request could not be completed.";
    if (error?.code !== "invalid_data" || !Array.isArray(error.details)) return message;
    const details = error.details.slice(0, 5).map((item) => {
      const path = Array.isArray(item.location)
        ? item.location.filter((part) => part !== "body").join(".").slice(0, 200)
        : "";
      return `${path || "Export"}: ${String(item.message || "Invalid value").slice(0, 300)}`;
    });
    if (error.details.length > 5) details.push(`${error.details.length - 5} more validation errors.`);
    return [message, ...details].join("\n");
  }
  function clearFailure() {
    $("job-detail").classList.remove("failed");
    $("job-guidance").textContent = "";
    $("job-guidance").hidden = true;
  }
  function notice(message = "", error = false) {
    $("notice").textContent = message;
    $("notice").classList.toggle("error", error);
    $("notice").hidden = !message;
  }
  function clearReport() {
    $("report-preview").removeAttribute("src");
    $("report-preview").hidden = true;
    $("download-report").hidden = true;
    $("download-report").removeAttribute("href");
    clearComparison();
    $("comparison-controls").hidden = true;
    $("comparison-controls").open = false;
    comparisonGeneration++;
    comparisonOffset = 0;
    comparisonHasMore = false;
    comparisonItems = [];
    $("comparison-target").replaceChildren();
    $("comparison-target").disabled = true;
    $("compare").disabled = true;
    $("comparison-status").textContent = "";
  }
  function clearComparison() {
    $("comparison-preview").removeAttribute("src");
    $("comparison-preview").hidden = true;
    $("comparison-actions").hidden = true;
    $("download-comparison").removeAttribute("href");
  }
  function clearCancellation() {
    $("cancel").hidden = true;
    $("cancel").disabled = true;
  }
  function reset() {
    epoch++;
    fileEpoch++;
    clearTimeout(timer);
    clearReport();
    clearCancellation();
    clearFailure();
    historyGeneration++;
    historyOffset = 0;
    historyHasMore = false;
    historyQuery = historyState = "";
    historyItems = [];
    historyPendingOffset = null;
    $("history-search").value = $("history-state").value = "";
    $("history-clear").hidden = true;
    $("history-loading").hidden = true;
    $("history-list").setAttribute("aria-busy", "false");
    $("history-previous").disabled = true;
    $("history-next").disabled = true;
    $("history-range").textContent = "";
    $("history-pages").hidden = true;
    modelGeneration++;
    models = [];
    activeModelId = null;
    modelOffset = 0;
    modelHasMore = modelsReady = modelsLoading = false;
    $("upload-model").replaceChildren();
    $("rerun-model").replaceChildren();
    $("upload-model").value = $("rerun-model").value = "";
    $("rerun-controls").hidden = true;
    queueStatus = queueCheckPending = null;
    queueCheckedAt = -Infinity;
    session = current = snapshot = result = exportText = retryAction = null;
    busy = false;
    $("token").value = "";
    $("export-file").value = "";
    $("file-details").hidden = true;
    $("analysis-panel").hidden = true;
    $("workspace").hidden = true;
    $("account").hidden = true;
    $("login-panel").hidden = false;
    $("history-list").replaceChildren();
    $("result-summary").replaceChildren();
    $("analysis-identity").textContent = "";
    $("job-detail").textContent = "";
    $("file-name").textContent = "";
    $("file-summary").textContent = "";
    $("classify").disabled = true;
    updateActions();
  }
  async function request(path, { method = "GET", body = undefined, headers = {} } = {}) {
    const requestEpoch = epoch;
    const options = {
      method,
      credentials: "same-origin",
      redirect: "error",
      cache: "no-store",
      headers: { ...headers },
    };
    if (body !== undefined) {
      options.body = typeof body === "string" ? body : JSON.stringify(body);
      options.headers["Content-Type"] = "application/json";
    }
    if (!["GET", "HEAD"].includes(method) && session)
      options.headers["X-CSRF-Token"] = session.csrf_token;
    let response;
    try {
      response = await fetch(path, options);
    } catch {
      throw new Error("Could not connect to the service. Check your connection and try again.");
    }
    if (requestEpoch !== epoch)
      throw new Error("The workspace changed. Refresh the page to continue.");
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      if (requestEpoch !== epoch)
        throw new Error("The workspace changed. Refresh the page to continue.");
      if (response.status === 401 && path !== "/api/v1/session") {
        reset();
        throw new Error("Your session has expired. Sign in to reopen your saved analyses.");
      }
      throw new Error(requestError(data));
    }
    return response;
  }
  async function json(path, options) {
    const requestEpoch = epoch;
    const value = await (await request(path, options)).json();
    if (requestEpoch !== epoch)
      throw new Error("The workspace changed. Refresh the page to continue.");
    return value;
  }
  async function activate(value) {
    session = value;
    $("token").value = "";
    $("owner").textContent = value.principal.owner_id;
    $("login-panel").hidden = true;
    $("workspace").hidden = false;
    $("account").hidden = false;
    notice();
    void loadModels();
    await history();
    const selected = new URL(location.href).searchParams.get("analysis");
    if (selected && /^[0-9a-f-]{36}$/.test(selected)) await openAnalysis(selected);
  }
  function option(value, label) {
    const item = document.createElement("option");
    item.value = value;
    item.textContent = label;
    return item;
  }
  function fillModelSelect(name, preferred = null) {
    const select = $(name);
    const selected = preferred ?? select.value;
    const choices = models.map((model) => [model.model_version_id, model.model_version_id]);
    // A previously selected published version remains selectable while browsing another page.
    if (selected && selected !== "active" && !choices.some(([id]) => id === selected))
      choices.unshift([selected, selected]);
    if (activeModelId) choices.unshift(["active", `Active model · ${activeModelId}`]);
    select.replaceChildren(option("", "Choose a published model"));
    choices.forEach(([id, label]) => select.append(option(id, label)));
    select.value = choices.some(([id]) => id === selected)
      ? selected : activeModelId ? "active" : "";
  }
  function updateActions() {
    $("classify").disabled = busy || !session || !exportText || !modelsReady || !$("upload-model").value;
    $("export-file").disabled = busy;
    $("upload-model").disabled = busy || modelsLoading || !modelsReady;
    $("rerun-model").disabled = busy || modelsLoading || !modelsReady;
    const saved = current?.metadata;
    const terminal = saved && ["succeeded", "failed", "cancelled"].includes(saved.job.state);
    $("rerun-controls").hidden = !terminal || !saved.snapshot_id;
    $("rerun").disabled = busy || !terminal || !modelsReady || !$("rerun-model").value;
    ["models-previous", "rerun-models-previous"].forEach((id) => {
      $(id).hidden = modelOffset === 0;
      $(id).disabled = busy || modelsLoading;
    });
    ["models-next", "rerun-models-next"].forEach((id) => {
      $(id).hidden = !modelHasMore;
      $(id).disabled = busy || modelsLoading;
    });
    $("models-refresh").disabled = busy || modelsLoading;
  }
  function modelStatus(message) {
    $("model-status").textContent = $("rerun-model-status").textContent = message;
  }
  async function loadModels(offset = 0) {
    const workspace = epoch;
    const generation = ++modelGeneration;
    modelsLoading = true;
    modelStatus("Loading models…");
    updateActions();
    try {
      const data = await json(`/api/v1/models?published_only=true&limit=${MODEL_PAGE_SIZE}&offset=${offset}`);
      if (workspace !== epoch || generation !== modelGeneration || !session) return;
      models = data.items;
      activeModelId = data.active_model_version_id;
      modelOffset = offset;
      modelHasMore = models.length === MODEL_PAGE_SIZE;
      modelsReady = true;
      fillModelSelect("upload-model");
      fillModelSelect("rerun-model", $("rerun-model").value || current?.metadata?.job.model?.model_version_id);
      modelStatus(models.length || activeModelId ? "" : offset
        ? "No more models. Return to the previous page."
        : "No published models are available. Ask your administrator to publish a model.");
    } catch (error) {
      if (workspace === epoch && generation === modelGeneration) modelStatus(error.message);
    } finally {
      if (workspace === epoch && generation === modelGeneration) {
        modelsLoading = false;
        updateActions();
      }
    }
  }
  async function history(offset = historyOffset) {
    const generation = ++historyGeneration;
    const workspace = epoch;
    historyPendingOffset = offset;
    $("history-previous").disabled = $("history-next").disabled = true;
    $("history-loading").hidden = false;
    $("history-list").setAttribute("aria-busy", "true");
    try {
      // One lookahead item determines whether another bounded page exists.
      const query = new URLSearchParams({limit: HISTORY_PAGE_SIZE + 1, offset});
      if (historyQuery) query.set("q", historyQuery);
      if (historyState) query.set("state", historyState);
      const data = await json(`/api/v1/analyses?${query}`);
      if (generation !== historyGeneration || workspace !== epoch || !session) return;
      const items = data.items.slice(0, HISTORY_PAGE_SIZE);
      historyOffset = offset;
      historyHasMore = data.items.length > HISTORY_PAGE_SIZE;
      $("history-pages").hidden = offset === 0 && !historyHasMore;
      $("history-empty").hidden = items.length > 0;
      $("history-empty").textContent = offset > 0
        ? "No analyses on this page. Return to the previous page or refresh."
        : historyQuery || historyState ? "No analyses match these filters." : "No analyses yet.";
      $("history-range").textContent = items.length ? `${offset + 1}–${offset + items.length}` : "";
      historyItems = items;
      renderHistory();
    } catch (error) {
      if (generation === historyGeneration && workspace === epoch) throw error;
    } finally {
      if (generation === historyGeneration && workspace === epoch) {
        historyPendingOffset = null;
        $("history-loading").hidden = true;
        $("history-list").setAttribute("aria-busy", "false");
        $("history-previous").disabled = historyOffset === 0;
        $("history-next").disabled = !historyHasMore;
      }
    }
  }
  function renderHistory() {
    const focused = document.activeElement?.dataset.analysisId;
    const scrollTop = $("history-list").scrollTop;
    let focusRestored = false;
    $("history-list").replaceChildren();
    historyItems.forEach((item) => {
      const row = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.analysisId = item.analysis_id;
      button.title = [item.run_id, item.source_id, item.analysis_id].filter(Boolean).join(" · ");
      const name = document.createElement("strong");
      name.textContent = item.instrument_ids?.length ? item.instrument_ids.join(", ") : item.run_id || "Analysis";
      const model = document.createElement("span");
      model.textContent = item.job.model?.model_version_id || "Model unavailable";
      const run = document.createElement("span");
      run.className = "history-run";
      run.textContent = item.run_id ? `Run ${item.run_id}` : "";
      const date = document.createElement("time");
      date.dateTime = item.job.created_at;
      date.textContent = new Date(item.job.created_at).toLocaleString(undefined, {
        dateStyle: "medium", timeStyle: "short",
      });
      const state = document.createElement("span");
      state.className = "history-state";
      state.dataset.state = item.job.state;
      state.textContent = `${stateLabels[item.job.state] || item.job.state} · ${item.analysis_id.slice(0, 4)}…${item.analysis_id.slice(-4)}`;
      button.append(name, model, run, date, state);
      button.addEventListener("click", () => openAnalysis(item.analysis_id));
      row.append(button);
      $("history-list").append(row);
      if (item.analysis_id === focused) {
        button.focus();
        focusRestored = true;
      }
    });
    $("history-list").scrollTop = scrollTop;
    if (focused && !focusRestored) $("refresh-history").focus();
    markSelectedHistory();
  }
  function applyHistoryFilters(event) {
    event?.preventDefault();
    historyQuery = $("history-search").value.trim();
    historyState = $("history-state").value;
    historyOffset = 0;
    historyHasMore = false;
    historyItems = [];
    renderHistory();
    $("history-empty").hidden = true;
    $("history-pages").hidden = true;
    $("history-clear").hidden = !historyQuery && !historyState;
    notice();
    return history(0).catch((error) => notice(error.message, true));
  }
  function updateHistory(value, changed) {
    const index = historyItems.findIndex((item) => item.analysis_id === value.analysis_id);
    if (index >= 0 && historyItems[index].job.state !== value.job.state) {
      historyItems[index] = {
        ...historyItems[index], ...value, job: {...historyItems[index].job, ...value.job},
      };
      renderHistory();
    }
    // Refetch after a state transition so an older list response cannot undo it.
    if (changed && (historyState || historyPendingOffset !== null))
      history(historyPendingOffset ?? historyOffset).catch((error) => notice(error.message, true));
  }
  function markSelectedHistory() {
    for (const row of $("history-list").children) {
      const button = row.children[0];
      if (button.dataset.analysisId === current?.id) button.setAttribute("aria-current", "true");
      else button.removeAttribute("aria-current");
    }
  }
  async function chooseFile(file) {
    if (busy)
      throw new Error(
        "Wait for the current request to be accepted before choosing another export.",
      );
    const selection = ++fileEpoch;
    exportText = snapshot = null;
    $("classify").disabled = true;
    $("file-details").hidden = true;
    notice();
    if (!file) return;
    if (file.size > MAX_BYTES)
      throw new Error("The export exceeds 16 MiB. Choose an export within the supported limit.");
    if (file.size === 0) throw new Error("The selected file is empty.");
    try {
      const text = await file.text();
      if (selection !== fileEpoch || !session) return;
      let data;
      try {
        data = JSON.parse(text);
      } catch {
        throw new Error("Choose a valid JSON export.");
      }
      if (data?.schema_version !== "2.1")
        throw new Error("This file is not a supported experiment export.");
      const validated = await json("/api/v1/validate", { method: "POST", body: text });
      if (selection !== fileEpoch || !session) return;
      exportText = text;
      $("file-name").textContent = file.name;
      $("file-summary").textContent =
        `${validated.step_count.toLocaleString()} steps · ${validated.instrument_ids.join(", ")}`;
      $("file-details").hidden = false;
      updateActions();
    } catch (error) {
      if (selection === fileEpoch && session) throw error;
    }
  }
  function retry(action) {
    retryAction = action;
    $("retry").hidden = !action;
  }
  function selectedModel(name) {
    return $(name).value === "active" ? activeModelId : $(name).value;
  }
  async function submit() {
    if (busy || !exportText || !modelsReady || !$("upload-model").value) return;
    return submitRequest({
      snapshotId: snapshot?.snapshot_id || null,
      exportText,
      model: selectedModel("upload-model"),
      key: crypto.randomUUID(),
      workspace: epoch,
    });
  }
  async function rerun() {
    const saved = current?.metadata;
    if (busy || !saved?.snapshot_id || !modelsReady || !$("rerun-model").value
      || !["succeeded", "failed", "cancelled"].includes(saved.job.state)) return;
    return submitRequest({
      snapshotId: saved.snapshot_id,
      exportText: null,
      model: selectedModel("rerun-model"),
      key: crypto.randomUUID(),
      workspace: epoch,
    });
  }
  async function submitRequest(submission) {
    if (busy || !session || submission.workspace !== epoch) return;
    busy = true;
    const workspace = epoch;
    const selection = { id: null, request: 0, job: null, cancelling: false };
    current = selection;
    result = null;
    clearTimeout(timer);
    clearCancellation();
    clearFailure();
    updateActions();
    $("analysis-panel").hidden = false;
    $("analysis-title").textContent = "Analysis";
    $("analysis-identity").textContent = "";
    $("job-state").textContent = "";
    $("job-detail").textContent = submission.snapshotId
      ? "Starting a new analysis of the saved experiment." : "Uploading the experiment.";
    $("progress").hidden = false;
    $("result").hidden = true;
    retry(null);
    clearReport();
    notice();
    try {
      if (!submission.snapshotId) {
        const uploaded = await json("/api/v1/snapshots", { method: "POST", body: submission.exportText });
        submission.snapshotId = uploaded.snapshot_id;
        if (workspace === epoch && exportText === submission.exportText) snapshot = uploaded;
      }
      const analysis = await json("/api/v1/analyses", {
        method: "POST",
        body: { snapshot_id: submission.snapshotId, model_version_id: submission.model },
        headers: { "Idempotency-Key": submission.key },
      });
      if (current === selection) await openAnalysis(analysis.analysis_id);
      if (workspace === epoch) await history(0).catch((error) => notice(error.message, true));
    } catch (error) {
      if (current !== selection || workspace !== epoch) return;
      $("progress").hidden = true;
      $("job-detail").textContent = "The request was not confirmed. Retry to check or continue it.";
      // Retry the captured request, even if the visible model selector later changes.
      retry(() => submitRequest(submission));
      notice(error.message, true);
    } finally {
      if (workspace === epoch) {
        busy = false;
        updateActions();
      }
    }
  }
  async function openAnalysis(id) {
    clearTimeout(timer);
    clearReport();
    clearCancellation();
    clearFailure();
    // Object identity also rejects an old response after selecting A, B, then A again.
    const selection = { id, request: 0, job: null, cancelling: false };
    current = selection;
    $("rerun-model").value = "";
    updateActions();
    result = null;
    retry(null);
    $("result").hidden = true;
    $("analysis-panel").hidden = false;
    $("analysis-title").textContent = "Analysis";
    markSelectedHistory();
    $("analysis-identity").textContent = id;
    $("job-state").textContent = "";
    $("job-detail").textContent = "Loading the selected analysis.";
    $("progress").hidden = false;
    notice();
    const url = new URL(location.href);
    url.searchParams.set("analysis", id);
    window.history.replaceState(null, "", url);
    await poll(selection);
  }
  function queueWait(job) {
    const since = Date.parse(job.updated_at || job.created_at);
    const seconds = Math.max(0, Math.floor((Date.now() - since) / 1000));
    if (!Number.isFinite(seconds)) return "";
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
    return `${Math.floor(seconds / 3600)}h ${Math.floor(seconds % 3600 / 60)}m`;
  }
  function showQueueStatus(selection) {
    if (current !== selection || selection?.metadata?.job.state !== "queued") return;
    const job = selection.metadata.job;
    const duration = queueWait(job);
    const waiting = job.stage === "retry_wait" ? "Waiting to retry" : "Waiting to start";
    $("job-detail").textContent = `${waiting}${duration ? ` · ${duration}` : ""}. `
      + (queueStatus?.message || "Checking worker availability…");
  }
  function checkQueueStatus(selection) {
    showQueueStatus(selection);
    if (queueCheckPending || Date.now() - queueCheckedAt < QUEUE_CHECK_INTERVAL) return;
    const workspace = epoch;
    const check = {};
    queueCheckPending = check;
    queueCheckedAt = Date.now();
    // Queue diagnostics are advisory; normal polling and result loading remain independent.
    void json("/api/v1/service-status").then((value) => {
      if (workspace === epoch) queueStatus = value;
    }).catch(() => {
      if (workspace === epoch)
        queueStatus = {message: "Worker availability could not be checked."};
    }).finally(() => {
      if (queueCheckPending === check) queueCheckPending = null;
      if (workspace === epoch) showQueueStatus(selection);
    });
  }
  async function poll(selection) {
    if (current !== selection || !session) return;
    clearTimeout(timer);
    const requestNumber = ++selection.request;
    const id = selection.id;
    const active = () => current === selection && requestNumber === selection.request;
    try {
      const value = await json(`/api/v1/analyses/${id}`);
      if (!active()) return;
      const job = value.job;
      if (!selection.metadata && modelsReady)
        fillModelSelect("rerun-model", job.model?.model_version_id);
      selection.metadata = value;
      updateActions();
      $("analysis-title").textContent = value.instrument_ids?.length
        ? value.instrument_ids.join(", ") : "Analysis";
      $("analysis-identity").textContent =
        `${id} · Model ${job.model?.model_version_id || "pending"}` +
        (value.run_id ? ` · Run ${value.run_id}` : "");
      $("job-state").textContent =
        job.state === "succeeded" ? "Complete" : job.state.replaceAll("_", " ");
      const pending = ["queued", "running", "cancel_requested"].includes(job.state);
      $("progress").hidden = !pending;
      selection.job = pending && job.state !== "cancel_requested" ? job.operation_id : null;
      $("cancel").hidden = selection.job === null;
      $("cancel").disabled = selection.cancelling || selection.job === null;
      $("job-detail").textContent =
        job.failure?.message ||
        (job.state === "failed"
          ? "The analysis failed before a result could be saved."
          : job.state === "succeeded"
          ? ""
          : job.state === "cancelled"
            ? "This analysis was cancelled."
            : job.state === "cancel_requested"
              ? "Cancelling the analysis."
              : stages[job.stage] || "Processing the analysis.");
      const failed = job.state === "failed";
      $("job-detail").classList.toggle("failed", failed);
      $("job-guidance").hidden = !failed;
      $("job-guidance").textContent = failed
        ? failureActions[job.failure?.code] || "Contact your administrator with the analysis ID under Details."
        : "";
      if (job.state === "queued") checkQueueStatus(selection);
      if (failed) {
        const attempts = Number.isInteger(job.attempt) ? ` · Attempts ${job.attempt}/${job.max_attempts}` : "";
        $("analysis-identity").textContent +=
          ` · Failure ${job.failure?.code || "unknown"}${attempts}`;
      }
      const stateChanged = selection.state !== job.state;
      selection.state = job.state;
      updateHistory({...value, analysis_id: id}, stateChanged);
      if (pending) timer = setTimeout(() => poll(selection), 1500);
      if (job.state === "succeeded" && value.result_available)
        await showResult(selection, value, active);
    } catch (error) {
      if (!active()) return;
      $("progress").hidden = true;
      retry(() => poll(selection));
      notice(error.message, true);
    }
  }
  async function showResult(selection, metadata, active) {
    const id = selection.id;
    const data = await json(`/api/v1/analyses/${id}/result`);
    if (!active()) return;
    result = data;
    $("result-summary").replaceChildren();
    [
      ["Instruments", data.instruments.map((item) => item.instrument_id).join(", ")],
      ["Time steps", data.time_grid.step_count],
    ].forEach(([name, value]) => {
      const item = document.createElement("div");
      const term = document.createElement("dt");
      term.textContent = name;
      const description = document.createElement("dd");
      description.textContent = String(value);
      item.append(term, description);
      $("result-summary").append(item);
    });
    $("result").hidden = false;
    if (!metadata.report_available) {
      $("job-detail").textContent =
        "The HTML report is unavailable. You can download the JSON result.";
    } else {
      clearReport();
      $("download-report").href = `/api/v1/analyses/${id}/report`;
      $("download-report").download = `regimex-analysis-${id}.html`;
      $("download-report").hidden = false;
      $("report-preview").src = `/api/v1/analyses/${id}/report/preview`;
      $("report-preview").hidden = false;
    }
    $("comparison-controls").hidden = !metadata.snapshot_id;
  }
  async function comparisonPage(offset = 0) {
    const selection = current;
    if (!session || selection?.metadata?.job.state !== "succeeded"
      || !selection.metadata.snapshot_id) return;
    const generation = ++comparisonGeneration;
    const workspace = epoch;
    const active = () => workspace === epoch && current === selection
      && generation === comparisonGeneration;
    $("comparison-target").disabled = $("compare").disabled = true;
    $("comparison-previous").disabled = $("comparison-next").disabled = true;
    $("comparison-status").textContent = "Loading completed analyses…";
    try {
      const query = new URLSearchParams({
        snapshot_id: selection.metadata.snapshot_id,
        state: "succeeded", limit: COMPARISON_PAGE_SIZE + 1, offset,
      });
      const data = await json(`/api/v1/analyses?${query}`);
      if (!active()) return;
      comparisonOffset = offset;
      comparisonHasMore = data.items.length > COMPARISON_PAGE_SIZE;
      comparisonItems = data.items.slice(0, COMPARISON_PAGE_SIZE).filter((item) =>
        item.analysis_id !== selection.id && item.snapshot_id === selection.metadata.snapshot_id
        && item.job.state === "succeeded" && item.result_available);
      const picker = $("comparison-target");
      picker.replaceChildren(option("", "Choose an analysis"));
      comparisonItems.forEach((item) => {
        const date = new Date(item.job.created_at).toLocaleString();
        picker.append(option(item.analysis_id,
          `${item.job.model?.model_version_id || "Model unavailable"} · ${date} · ${item.analysis_id.slice(-8)}`));
      });
      picker.value = "";
      $("comparison-status").textContent = comparisonItems.length ? "" : offset || comparisonHasMore
        ? "No other completed analyses on this page."
        : "No other completed analyses of this experiment yet. Run it again to compare results.";
      $("comparison-previous").hidden = offset === 0;
      $("comparison-next").hidden = !comparisonHasMore;
      picker.disabled = comparisonItems.length === 0;
      selection.comparisonLoaded = true;
    } catch (error) {
      if (active()) $("comparison-status").textContent = error.message;
    } finally {
      if (active()) {
        $("comparison-previous").disabled = comparisonOffset === 0;
        $("comparison-next").disabled = !comparisonHasMore;
      }
    }
  }
  function compare() {
    const selection = current;
    const other = comparisonItems.find((item) => item.analysis_id === $("comparison-target").value);
    if (!selection?.metadata || !other || other.snapshot_id !== selection.metadata.snapshot_id
      || selection.metadata.job.state !== "succeeded") return;
    const path = `/api/v1/analyses/${selection.id}/comparison`;
    const query = `other=${encodeURIComponent(other.analysis_id)}`;
    $("comparison-preview").src = `${path}/preview?${query}`;
    $("comparison-preview").hidden = false;
    $("report-preview").hidden = true;
    $("download-comparison").href = `${path}?${query}`;
    $("download-comparison").download = `regimex-comparison-${selection.id}-${other.analysis_id}.html`;
    $("comparison-actions").hidden = false;
  }
  $("login-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    $("login-submit").disabled = true;
    const token = $("token").value.trim();
    $("token").value = "";
    try {
      await activate(await json("/api/v1/session", { method: "POST", body: { token } }));
    } catch (error) {
      notice(error.message, true);
    } finally {
      $("login-submit").disabled = false;
    }
  });
  $("logout").addEventListener("click", async () => {
    try {
      await request("/api/v1/session", { method: "DELETE" });
      reset();
      const url = new URL(location.href);
      url.searchParams.delete("analysis");
      window.history.replaceState(null, "", url);
      notice();
    } catch (error) {
      notice(error.message, true);
    }
  });
  $("export-file").addEventListener("change", (event) =>
    chooseFile(event.target.files[0]).catch((error) => notice(error.message, true)),
  );
  ["dragenter", "dragover"].forEach((name) =>
    $("drop-zone").addEventListener(name, (event) => {
      event.preventDefault();
      $("drop-zone").classList.add("dragging");
    }),
  );
  ["dragleave", "drop"].forEach((name) =>
    $("drop-zone").addEventListener(name, (event) => {
      event.preventDefault();
      $("drop-zone").classList.remove("dragging");
    }),
  );
  $("drop-zone").addEventListener("drop", (event) => {
    if (event.dataTransfer.files.length !== 1) {
      notice("Choose one completed-run export.", true);
      return;
    }
    chooseFile(event.dataTransfer.files[0]).catch((error) => notice(error.message, true));
  });
  $("classify").addEventListener("click", submit);
  $("rerun").addEventListener("click", rerun);
  $("upload-model").addEventListener("change", updateActions);
  $("rerun-model").addEventListener("change", updateActions);
  ["models-previous", "rerun-models-previous"].forEach((id) =>
    $(id).addEventListener("click", () => loadModels(Math.max(0, modelOffset - MODEL_PAGE_SIZE))));
  ["models-next", "rerun-models-next"].forEach((id) =>
    $(id).addEventListener("click", () => loadModels(modelOffset + MODEL_PAGE_SIZE)));
  $("models-refresh").addEventListener("click", () => loadModels());
  $("comparison-controls").addEventListener("toggle", () => {
    if ($("comparison-controls").open && !current?.comparisonLoaded) return comparisonPage();
  });
  $("comparison-target").addEventListener("change", () => {
    $("compare").disabled = !comparisonItems.some((item) => item.analysis_id === $("comparison-target").value);
  });
  $("comparison-previous").addEventListener("click", () =>
    comparisonPage(Math.max(0, comparisonOffset - COMPARISON_PAGE_SIZE)));
  $("comparison-next").addEventListener("click", () =>
    comparisonPage(comparisonOffset + COMPARISON_PAGE_SIZE));
  $("comparison-refresh").addEventListener("click", () => comparisonPage());
  $("compare").addEventListener("click", compare);
  $("close-comparison").addEventListener("click", () => {
    clearComparison();
    $("report-preview").hidden = !current?.metadata?.report_available;
  });
  $("retry").addEventListener("click", () => {
    notice();
    const action = retryAction;
    retry(null);
    if (action) action().catch((error) => notice(error.message, true));
  });
  $("cancel").addEventListener("click", async () => {
    const selection = current;
    if (!selection?.job || selection.cancelling || !session) return;
    const job = selection.job;
    selection.cancelling = true;
    selection.request++;
    clearTimeout(timer);
    $("cancel").disabled = true;
    try {
      await json(`/api/v1/jobs/${job}/cancel`, { method: "POST" });
      if (current === selection) await poll(selection);
    } catch (error) {
      if (current === selection) {
        notice(error.message, true);
        timer = setTimeout(() => poll(selection), 1500);
      }
    } finally {
      selection.cancelling = false;
      if (current === selection) $("cancel").disabled = selection.job === null;
    }
  });
  $("refresh-history").addEventListener("click", () =>
    history(0).catch((error) => notice(error.message, true)),
  );
  $("history-filters").addEventListener("submit", applyHistoryFilters);
  $("history-state").addEventListener("change", applyHistoryFilters);
  $("history-clear").addEventListener("click", () => {
    $("history-search").value = $("history-state").value = "";
    return applyHistoryFilters();
  });
  $("history-previous").addEventListener("click", () =>
    history(Math.max(0, historyOffset - HISTORY_PAGE_SIZE)).catch((error) =>
      notice(error.message, true),
    ),
  );
  $("history-next").addEventListener("click", () =>
    history(historyOffset + HISTORY_PAGE_SIZE).catch((error) => notice(error.message, true)),
  );
  $("download-json").addEventListener("click", () => {
    if (!result) return;
    const link = document.createElement("a");
    link.href = `/api/v1/analyses/${current.id}/result`;
    link.download = `regimex-analysis-${current.id}.json`;
    link.click();
  });
  window.addEventListener("pagehide", clearReport);
  json("/api/v1/session")
    .then(activate)
    .catch((error) => {
      reset();
      if (!error.message.includes("credential") && !error.message.includes("session"))
        notice(error.message, true);
    });
})();
