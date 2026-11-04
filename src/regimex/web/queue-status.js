import {$} from "./ui.js";

const QUEUE_CHECK_INTERVAL = 5000;

export function createQueueStatus({session, getSelection}) {
  let queueStatus = null;
  let queueCheckedAt = -Infinity;
  let queueCheckPending = null;
  function reset() {
    queueStatus = queueCheckPending = null;
    queueCheckedAt = -Infinity;
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
    if (getSelection() !== selection || selection?.metadata?.job.state !== "queued") return;
    const job = selection.metadata.job;
    const duration = queueWait(job);
    const waiting = job.stage === "retry_wait" ? "Waiting to retry" : "Waiting to start";
    $("job-detail").textContent = `${waiting}${duration ? ` · ${duration}` : ""}. `
      + (queueStatus?.message || "Checking worker availability…");
  }

  function checkQueueStatus(selection) {
    showQueueStatus(selection);
    if (queueCheckPending || Date.now() - queueCheckedAt < QUEUE_CHECK_INTERVAL) return;
    const workspace = session.epoch;
    const check = {};
    queueCheckPending = check;
    queueCheckedAt = Date.now();
    // Queue diagnostics are advisory; normal polling and result loading remain independent.
    void session.json("/api/v1/service-status").then((value) => {
      if (workspace === session.epoch) queueStatus = value;
    }).catch(() => {
      if (workspace === session.epoch)
        queueStatus = {message: "Worker availability could not be checked."};
    }).finally(() => {
      if (queueCheckPending === check) queueCheckPending = null;
      if (workspace === session.epoch) showQueueStatus(selection);
    });
  }
  return {reset, check: checkQueueStatus};
}
