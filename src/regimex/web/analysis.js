import {$, notice} from "./ui.js";
import {createQueueStatus} from "./queue-status.js";

const MAX_BYTES = 16 * 1024 * 1024;
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

export function createAnalysis({session, models, history, comparison}) {
  let exportText = null;
  let snapshot = null;
  let current = null;
  let result = null;
  let timer = null;
  let retryAction = null;
  let busy = false;
  let fileEpoch = 0;
  const queue = createQueueStatus({session, getSelection: () => current});

  function clearReport() {
    $("report-preview").removeAttribute("src");
    $("report-preview").hidden = true;
    $("download-report").hidden = true;
    $("download-report").removeAttribute("href");
    comparison.reset();
  }
  function reset() {
    fileEpoch++;
    clearTimeout(timer);
    clearReport();
    clearCancellation();
    clearFailure();
    queue.reset();
    current = snapshot = result = exportText = retryAction = null;
    busy = false;
    $("export-file").value = "";
    $("file-details").hidden = true;
    $("analysis-panel").hidden = true;
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
      if (selection !== fileEpoch || !session.value) return;
      let data;
      try {
        data = JSON.parse(text);
      } catch {
        throw new Error("Choose a valid JSON export.");
      }
      if (data?.schema_version !== "2.1")
        throw new Error("This file is not a supported experiment export.");
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
    if (busy || !exportText || !models.ready || !$("upload-model").value) return;
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
      const analysis = await session.json("/api/v1/analyses", {
        method: "POST",
        body: { snapshot_id: submission.snapshotId, model_version_id: submission.model },
        headers: { "Idempotency-Key": submission.key },
      });
      if (current === selection) await openAnalysis(analysis.analysis_id);
      if (workspace === session.epoch) await history.load(0).catch((error) => notice(error.message, true));
    } catch (error) {
      if (current !== selection || workspace !== session.epoch) return;
      $("progress").hidden = true;
      $("job-detail").textContent = "The request was not confirmed. Retry to check or continue it.";
      // Retry the captured request, even if the visible model selector later changes.
      retry(() => submitRequest(submission));
      notice(error.message, true);
    } finally {
      if (workspace === session.epoch) {
        busy = false;
        models.updateActions();
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
    const url = new URL(location.href);
    url.searchParams.set("analysis", id);
    window.history.replaceState(null, "", url);
    await poll(selection);
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
      if (job.state === "queued") queue.check(selection);
      if (failed) {
        const attempts = Number.isInteger(job.attempt) ? ` · Attempts ${job.attempt}/${job.max_attempts}` : "";
        $("analysis-identity").textContent +=
          ` · Failure ${job.failure?.code || "unknown"}${attempts}`;
      }
      const stateChanged = selection.state !== job.state;
      selection.state = job.state;
      history.update({...value, analysis_id: id}, stateChanged);
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
    const data = await session.json(`/api/v1/analyses/${id}/result`);
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
  window.addEventListener("pagehide", clearReport);
  return {reset, open: openAnalysis, get current() { return current; },
    get busy() { return busy; }, get exportText() { return exportText; }};
}
