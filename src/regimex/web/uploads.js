import {$, analysisStateLabel} from "./ui.js";

const MAX_BYTES = 16 * 1024 * 1024;
const ACTIVE_LIMIT = 2;
const POLL_MS = 2000;

export async function readExport(file) {
  if (file.size > MAX_BYTES)
    throw new Error("The export exceeds 16 MiB. Choose an export within the supported limit.");
  if (!file.size) throw new Error("The selected file is empty.");
  const text = await file.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error("Choose a valid JSON export.");
  }
  if (data?.schema_version !== "2.1")
    throw new Error("This file is not a supported experiment export.");
  return text;
}

export function requestAnalysis(session, snapshotId, model, key) {
  return session.json("/api/v1/analyses", {
    method: "POST",
    body: {snapshot_id: snapshotId, model_version_id: model},
    headers: {"Idempotency-Key": key},
  });
}

function fileRow(file, openAnalysis, retryFailed) {
  const row = document.createElement("li");
  const name = document.createElement("strong");
  name.textContent = file.name;
  const detail = document.createElement("span");
  detail.textContent = "Ready";
  const open = document.createElement("button");
  open.type = "button";
  open.className = "quiet";
  open.textContent = "Open analysis";
  open.setAttribute("aria-label", `Open analysis for ${file.name}`);
  open.hidden = true;
  const retry = document.createElement("button");
  retry.type = "button";
  retry.className = "quiet";
  retry.textContent = "Retry analysis";
  retry.setAttribute("aria-label", `Retry analysis for ${file.name}`);
  retry.hidden = true;
  const entry = {file, phase: "pending", snapshotId: null, analysisId: null,
    key: crypto.randomUUID(), detail, open, retry};
  open.addEventListener("click", () => openAnalysis(entry.analysisId));
  retry.addEventListener("click", () => retryFailed(entry));
  row.append(name, detail, open, retry);
  $("batch-files").append(row);
  return entry;
}

export function createBatch({session, openAnalysis, updateActions, refreshHistory}) {
  let current = null;
  let timer = null;

  function clear() {
    current = null;
    clearTimeout(timer);
    $("batch-panel").hidden = true;
    $("batch-files").replaceChildren();
    $("batch-status").textContent = "";
    $("classify").hidden = false;
    $("classify").textContent = "Analyze";
    updateActions();
  }
  function choose(files) {
    current = {rows: files.map(file => fileRow(file, openAnalysis, retryFailed)), model: null, workspace: session.epoch,
      paused: "", processing: false, resumeAfterCompletion: false};
    $("batch-panel").hidden = false;
    $("classify").textContent = `Analyze ${files.length} files`;
    render(current);
  }
  function render(batch) {
    const pending = batch.rows.filter(row => row.phase === "pending").length;
    const active = batch.rows.filter(row => row.phase === "active").length;
    const finished = batch.rows.length - pending - active;
    for (const row of batch.rows) {
      row.retry.hidden = row.phase !== "failed";
      row.retry.disabled = Boolean(batch.paused);
    }
    $("classify").hidden = Boolean(batch.model);
    $("batch-retry").hidden = Boolean(batch.paused || pending || active)
      || !batch.rows.some(row => row.phase === "failed");
    $("batch-resume").hidden = !batch.paused;
    const progress = batch.model
      ? `${finished} of ${batch.rows.length} finished · ${batch.model}.`
      : `${batch.rows.length} files selected.`;
    const next = batch.paused || (batch.model && pending
      ? "Keep this batch open to submit the remaining files." : "");
    $("batch-status").textContent = [progress, next].filter(Boolean).join(" ");
    updateActions();
  }
  function pause(batch, error, mayResume = false) {
    batch.paused = error.message;
    batch.resumeAfterCompletion = mayResume;
  }
  async function submit(batch, row) {
    if (!row.snapshotId) {
      let text;
      try {
        text = await readExport(row.file);
      } catch (error) {
        row.phase = "invalid";
        row.detail.textContent = error.message;
        return;
      }
      if (current !== batch) return;
      row.detail.textContent = "Uploading";
      const uploaded = await session.json("/api/v1/snapshots", {method: "POST", body: text});
      row.snapshotId = uploaded.snapshot_id;
    }
    if (current !== batch) return;
    row.detail.textContent = "Submitting";
    const accepted = await requestAnalysis(session, row.snapshotId, batch.model, row.key);
    row.analysisId = accepted.analysis_id;
    row.phase = "active";
    row.detail.textContent = "Queued";
    row.open.hidden = false;
  }
  async function submitPending(batch) {
    let changed = false;
    const active = batch.rows.filter(row => row.phase === "active").length;
    const ready = batch.rows.filter(row => row.phase === "pending").slice(0, ACTIVE_LIMIT - active);
    for (const row of ready) {
      try {
        await submit(batch, row);
        changed ||= Boolean(row.snapshotId);
      } catch (error) {
        row.detail.textContent = error.message;
        if (!error.status || error.status === 429 || error.status >= 500) {
          // The server may already have accepted an unconfirmed request. Keep its key.
          pause(batch, error, error.status === 429 && Boolean(row.snapshotId));
        } else row.phase = "invalid";
        changed ||= Boolean(row.snapshotId);
      }
      if (current !== batch || batch.paused) break;
    }
    return changed;
  }
  async function pollAccepted(batch) {
    let changed = false;
    for (const row of batch.rows.filter(row => row.phase === "active")) {
      try {
        const {job} = await session.json(`/api/v1/analyses/${row.analysisId}`);
        if (current !== batch || row.phase !== "active") continue;
        row.detail.textContent = job.failure?.message
          ? `${analysisStateLabel(job.state)} · ${job.failure.message}` : analysisStateLabel(job.state);
        if (["succeeded", "failed", "cancelled"].includes(job.state)) {
          row.phase = job.state;
          changed = true;
          if (batch.resumeAfterCompletion) {
            batch.paused = "";
            batch.resumeAfterCompletion = false;
          }
        }
      } catch (error) {
        if (current !== batch) return changed;
        row.detail.textContent = error.message;
        if ([403, 404].includes(error.status)) {
          row.phase = "unavailable";
          row.open.hidden = true;
        } else {
          pause(batch, error);
          break;
        }
      }
    }
    return changed;
  }
  async function advance(batch) {
    if (current !== batch || batch.processing) return;
    clearTimeout(timer);
    batch.processing = true;
    const submitted = !batch.paused && await submitPending(batch);
    if (current !== batch) {
      if (submitted && batch.workspace === session.epoch) refreshHistory();
      return;
    }
    const completed = await pollAccepted(batch);
    if (current !== batch) return;
    batch.processing = false;
    render(batch);
    if (submitted || completed) refreshHistory();
    const active = batch.rows.filter(row => row.phase === "active").length;
    const pending = batch.rows.some(row => row.phase === "pending");
    if (batch.paused && !(batch.resumeAfterCompletion && active)) return;
    if (active || pending)
      timer = setTimeout(() => advance(batch), !batch.paused && pending && active < ACTIVE_LIMIT ? 0 : POLL_MS);
  }
  function start(model) {
    if (!current || current.model || !model) return;
    current.model = model;
    render(current);
    return advance(current);
  }
  $("batch-resume").addEventListener("click", () => {
    if (!current?.paused) return;
    current.paused = "";
    current.resumeAfterCompletion = false;
    render(current);
    return advance(current);
  });
  function retryFailed(only = null) {
    if (!current || current.paused) return;
    const rows = current.rows.filter(row => row.phase === "failed" && (!only || row === only));
    if (!rows.length) return;
    for (const row of rows) {
      // A confirmed terminal failure needs a new analysis of the saved snapshot.
      row.key = crypto.randomUUID();
      row.analysisId = null;
      row.phase = "pending";
      row.open.hidden = true;
      row.detail.textContent = "Ready";
    }
    render(current);
    return advance(current);
  }
  $("batch-retry").addEventListener("click", () => retryFailed());
  $("batch-close").addEventListener("click", () => {
    $("export-file").value = "";
    clear();
  });
  function forgetDeleted(target) {
    if (!current) return;
    for (const row of current.rows) {
      const matches = target.kind === "analysis" ? row.analysisId === target.id
        : row.snapshotId === target.id;
      if (matches) {
        row.phase = "deleted";
        row.open.hidden = true;
        row.detail.textContent = "Deleted";
      }
    }
    render(current);
  }
  return {choose, clear, forgetDeleted, start, get selected() { return Boolean(current); },
    get ready() { return Boolean(current && !current.model); },
    get locked() { return Boolean(current?.model); }};
}
