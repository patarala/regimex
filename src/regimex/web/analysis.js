import {$, analysisStateLabel, notice, setPreview} from "./ui.js";
import {createQueueStatus} from "./queue-status.js";
import {createBatch, readExport, requestAnalysis} from "./uploads.js";

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

export function createAnalysis({session, models, history, comparison, onStorageChange,
  onSelectionChange, rememberAnalysis, deleteAnalysis}) {
  let exportText = null;
  let snapshot = null;
  let current = null;
  let result = null;
  let timer = null;
  let retryAction = null;
  let busy = false;
  let fileEpoch = 0;
  const queue = createQueueStatus({session, getSelection: () => current});
  const batch = createBatch({session, openAnalysis, updateActions: models.updateActions,
    refreshHistory: () => { void history.load(0); onStorageChange(); }});

  function clearReport() {
    setPreview("report-preview");
    $("download-report").hidden = true;
    $("download-report").removeAttribute("href");
    comparison.reset();
  }
  function reset() {
    onSelectionChange();
    fileEpoch++;
    clearTimeout(timer);
    clearReport();
    clearCancellation();
    clearFailure();
    queue.reset();
    current = snapshot = result = exportText = retryAction = null;
    batch.clear();
    busy = false;
    $("export-file").value = "";
    $("file-details").hidden = true;
    $("analysis-panel").hidden = true;
    $("delete-analysis").hidden = true;
    $("result-summary").replaceChildren();
    $("analysis-identity").textContent = "";
    $("job-detail").textContent = "";
    $("file-name").textContent = "";
    $("file-summary").textContent = "";
    models.updateActions();
  }

  function clearFailure() {
    $("job-detail").classList.remove("failed");
    $("job-guidance").textContent = "";
    $("job-guidance").hidden = true;
  }

  function clearCancellation() {
    $("cancel").hidden = true;
    $("cancel").disabled = true;
  }

  async function chooseFiles(files) {
    if (busy || batch.locked)
      throw new Error(
        batch.locked ? "Close the current batch before choosing more files."
          : "Wait for the current request to be accepted before choosing another export.",
      );
    const selected = Array.from(files);
    const selection = ++fileEpoch;
    exportText = snapshot = null;
    batch.clear();
    $("classify").disabled = true;
    $("file-details").hidden = true;
    notice();
    if (selected.length > 1) {
      batch.choose(selected);
      return;
    }
    const [file] = selected;
    if (!file) return;
    try {
      const text = await readExport(file);
      if (selection !== fileEpoch || !session.value) return;
      const validated = await session.json("/api/v1/validate", { method: "POST", body: text });
      if (selection !== fileEpoch || !session.value) return;
      exportText = text;
      $("file-name").textContent = file.name;
      $("file-summary").textContent =
        `${validated.step_count.toLocaleString()} steps · ${validated.instrument_ids.join(", ")}`;
      $("file-details").hidden = false;
      models.updateActions();
    } catch (error) {
      if (selection === fileEpoch && session.value) throw error;
    }
  }

  function retry(action) {
    retryAction = action;
    $("retry").hidden = !action;
  }

  async function submit() {
    if (busy || !models.ready || !$("upload-model").value) return;
    if (batch.selected) return batch.start(models.selected("upload-model"));
    if (!exportText) return;
    return submitRequest({
      snapshotId: snapshot?.snapshot_id || null,
      exportText,
      model: models.selected("upload-model"),
      key: crypto.randomUUID(),
      workspace: session.epoch,
    });
  }

  async function rerun() {
    const saved = current?.metadata;
    if (busy || !saved?.snapshot_id || !models.ready || !$("rerun-model").value
      || !["succeeded", "failed", "cancelled"].includes(saved.job.state)) return;
    return submitRequest({
      snapshotId: saved.snapshot_id,
      exportText: null,
      model: models.selected("rerun-model"),
      key: crypto.randomUUID(),
      workspace: session.epoch,
    });
  }

  async function submitRequest(submission) {
    if (busy || !session.value || submission.workspace !== session.epoch) return;
    busy = true;
    onSelectionChange();
    $("delete-analysis").hidden = true;
    const workspace = session.epoch;
    const selection = { id: null, request: 0, job: null, cancelling: false };
    current = selection;
    result = null;
    clearTimeout(timer);
    clearCancellation();
    clearFailure();
    models.updateActions();
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
        const uploaded = await session.json("/api/v1/snapshots", { method: "POST", body: submission.exportText });
        submission.snapshotId = uploaded.snapshot_id;
        if (workspace === session.epoch && exportText === submission.exportText) snapshot = uploaded;
      }
      if (workspace !== session.epoch) return;
      const analysis = await requestAnalysis(session, submission.snapshotId, submission.model, submission.key);
      if (current === selection) await openAnalysis(analysis.analysis_id);
      if (workspace === session.epoch) await history.load(0);
    } catch (error) {
      if (current !== selection || workspace !== session.epoch) return;
      $("progress").hidden = true;
      $("job-detail").textContent = "The request was not confirmed. Retry to check or continue it.";
      // A retry must send the same model version and idempotency key.
      retry(() => submitRequest(submission));
      notice(error.message, true);
    } finally {
      if (workspace === session.epoch) {
        onStorageChange();
        busy = false;
        models.updateActions();
      }
    }
  }

  async function openAnalysis(id, {restore = false} = {}) {
    onSelectionChange();
    $("delete-analysis").hidden = true;
    clearTimeout(timer);
    clearReport();
    clearCancellation();
    clearFailure();
    // Object identity also rejects an old response after selecting A, B, then A again.
    const selection = { id, request: 0, job: null, cancelling: false };
    current = selection;
    $("rerun-model").value = "";
    models.updateActions();
    result = null;
    retry(null);
    $("result").hidden = true;
    $("analysis-panel").hidden = false;
    $("analysis-title").textContent = "Analysis";
    history.markSelected();
    $("analysis-identity").textContent = id;
    $("job-state").textContent = "";
    $("job-detail").textContent = "Loading the selected analysis.";
    $("progress").hidden = false;
    notice();
    if (!restore) rememberAnalysis(id);
    selection.loading = poll(selection);
    await selection.loading;
    selection.loading = null;
    return selection;
  }

  async function poll(selection) {
    if (current !== selection || !session.value) return;
    clearTimeout(timer);
    const requestNumber = ++selection.request;
    const id = selection.id;
    const active = () => current === selection && requestNumber === selection.request;
    try {
      const value = await session.json(`/api/v1/analyses/${id}`);
      if (!active()) return;
      const job = value.job;
      if (!selection.metadata && models.ready)
        models.fillSelect("rerun-model", job.model?.model_version_id);
      selection.metadata = value;
      models.updateActions();
      $("analysis-title").textContent = value.instrument_ids?.length
        ? value.instrument_ids.join(", ") : "Analysis";
      $("analysis-identity").textContent =
        `${id} · Model ${job.model?.model_version_id || "pending"}` +
        (value.run_id ? ` · Run ${value.run_id}` : "");
      $("job-state").textContent = analysisStateLabel(job.state);
      $("job-state").dataset.state = job.state;
      const pending = ["queued", "running", "cancel_requested"].includes(job.state);
      $("delete-analysis").hidden = pending;
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
      if (job.state === "queued") queue.check(selection);
      if (failed) {
        const attempts = Number.isInteger(job.attempt) ? ` · Attempts ${job.attempt}/${job.max_attempts}` : "";
        $("analysis-identity").textContent +=
          ` · Failure ${job.failure?.code || "unknown"}${attempts}`;
      }
      const stateChanged = selection.state !== job.state;
      selection.state = job.state;
      history.update({...value, analysis_id: id}, stateChanged);
      if (stateChanged && !pending) onStorageChange();
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
    let instruments = metadata.instrument_ids;
    let steps = metadata.step_count;
    if (!instruments?.length || steps == null) {
      // Snapshots created before grid summaries were stored only carry this in the result.
      const data = await session.json(`/api/v1/analyses/${id}/result`);
      if (!active()) return;
      instruments = data.instruments.map((item) => item.instrument_id);
      steps = data.time_grid.step_count;
    }
    result = metadata.result_available;
    $("result-summary").replaceChildren();
    [
      ["Instruments", instruments.join(", ")],
      ["Time steps", steps],
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
      setPreview("report-preview", `/api/v1/analyses/${id}/report/preview`);
    }
    $("comparison-controls").hidden = !metadata.snapshot_id;
  }
  $("export-file").addEventListener("change", (event) =>
    chooseFiles(event.target.files).catch((error) => notice(error.message, true)),
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
    chooseFiles(event.dataTransfer.files).catch((error) => notice(error.message, true));
  });
  $("classify").addEventListener("click", submit);
  $("rerun").addEventListener("click", rerun);
  $("retry").addEventListener("click", () => {
    notice();
    const action = retryAction;
    retry(null);
    if (action) action().catch((error) => notice(error.message, true));
  });
  $("cancel").addEventListener("click", async () => {
    const selection = current;
    if (!selection?.job || selection.cancelling || !session.value) return;
    const job = selection.job;
    selection.cancelling = true;
    selection.request++;
    clearTimeout(timer);
    $("cancel").disabled = true;
    try {
      await session.json(`/api/v1/jobs/${job}/cancel`, { method: "POST" });
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
  $("download-json").addEventListener("click", () => {
    if (!result) return;
    const link = document.createElement("a");
    link.href = `/api/v1/analyses/${current.id}/result`;
    link.download = `regimex-analysis-${current.id}.json`;
    link.click();
  });
  $("delete-analysis").addEventListener("click", () => {
    if (busy || !current?.metadata || !["succeeded", "failed", "cancelled"].includes(current.metadata.job.state)) return;
    return deleteAnalysis(current.id, $("analysis-title").textContent);
  });
  function clearSelection() {
    clearTimeout(timer);
    clearReport();
    clearCancellation();
    clearFailure();
    queue.reset();
    current = result = retryAction = null;
    $("analysis-panel").hidden = true;
    $("delete-analysis").hidden = true;
    $("result-summary").replaceChildren();
    $("analysis-identity").textContent = "";
    models.updateActions();
    history.markSelected();
  }

  function forgetDeleted(target) {
    batch.forgetDeleted(target);
    if (target.kind === "experiment" && snapshot?.snapshot_id === target.id) snapshot = null;
    const affected = current && (target.kind === "analysis" ? current.id === target.id
      : current.metadata?.snapshot_id === target.id);
    if (affected) {
      clearSelection();
      rememberAnalysis(null);
    } else if (current) {
      comparison.reset();
      current.comparisonLoaded = false;
      $("comparison-controls").hidden = !current.metadata?.result_available;
      $("report-preview").hidden = !current.metadata?.report_available;
    }
  }
  return {reset, open: openAnalysis, clearSelection, forgetDeleted, get current() { return current; },
    get busy() { return busy; }, get uploadReady() { return Boolean(exportText) || batch.ready; },
    get batchLocked() { return batch.locked; }};
}
