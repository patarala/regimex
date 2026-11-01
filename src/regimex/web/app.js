"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const MAX_BYTES = 16 * 1024 * 1024;
  const HISTORY_PAGE_SIZE = 20;
  let session = null;
  let exportText = null;
  let snapshot = null;
  let intakeKey = null;
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
  const stages = {
    queued: "Waiting to start.",
    loading: "Loading the experiment.",
    predicting: "Analyzing prices.",
    rendering: "Preparing the report.",
    committing: "Saving the analysis.",
    retry_wait: "Waiting to retry.",
  };
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
    historyGeneration++;
    historyOffset = 0;
    historyHasMore = false;
    $("history-previous").disabled = true;
    $("history-next").disabled = true;
    $("history-range").textContent = "";
    $("history-pages").hidden = true;
    session = current = snapshot = result = exportText = intakeKey = retryAction = null;
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
      throw new Error(data.error?.message || "The request could not be completed.");
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
    await history();
    const selected = new URL(location.href).searchParams.get("analysis");
    if (selected && /^[0-9a-f-]{36}$/.test(selected)) await openAnalysis(selected);
  }
  async function history(offset = historyOffset) {
    const generation = ++historyGeneration;
    const workspace = epoch;
    $("history-previous").disabled = $("history-next").disabled = true;
    try {
      // One lookahead item determines whether another bounded page exists.
      const data = await json(`/api/v1/analyses?limit=${HISTORY_PAGE_SIZE + 1}&offset=${offset}`);
      if (generation !== historyGeneration || workspace !== epoch || !session) return;
      const items = data.items.slice(0, HISTORY_PAGE_SIZE);
      historyOffset = offset;
      historyHasMore = data.items.length > HISTORY_PAGE_SIZE;
      $("history-pages").hidden = offset === 0 && !historyHasMore;
      $("history-list").replaceChildren();
      $("history-empty").hidden = items.length > 0;
      $("history-range").textContent = items.length ? `${offset + 1}–${offset + items.length}` : "";
      items.forEach((item) => {
        const row = document.createElement("li");
        const button = document.createElement("button");
        button.type = "button";
        button.dataset.analysisId = item.analysis_id;
        button.title = item.analysis_id;
        const name = document.createElement("span");
        name.textContent = new Date(item.job.created_at).toLocaleString(undefined, {
          dateStyle: "medium",
          timeStyle: "short",
        });
        const state = document.createElement("span");
        state.textContent = `${item.job.state === "succeeded" ? "Complete" : item.job.state.replaceAll("_", " ")} · ${item.analysis_id.slice(0, 4)}…${item.analysis_id.slice(-4)}`;
        button.append(name, state);
        button.addEventListener("click", () => openAnalysis(item.analysis_id));
        row.append(button);
        $("history-list").append(row);
      });
      markSelectedHistory();
    } catch (error) {
      if (generation === historyGeneration && workspace === epoch) throw error;
    } finally {
      if (generation === historyGeneration && workspace === epoch) {
        $("history-previous").disabled = historyOffset === 0;
        $("history-next").disabled = !historyHasMore;
      }
    }
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
    exportText = snapshot = intakeKey = null;
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
      intakeKey = crypto.randomUUID();
      $("file-name").textContent = file.name;
      $("file-summary").textContent =
        `${validated.step_count.toLocaleString()} steps · ${validated.instrument_ids.join(", ")}`;
      $("file-details").hidden = false;
      $("classify").disabled = false;
    } catch (error) {
      if (selection === fileEpoch && session) throw error;
    }
  }
  function retry(action) {
    retryAction = action;
    $("retry").hidden = !action;
  }
  async function submit() {
    if (busy || !exportText) return;
    busy = true;
    const workspace = epoch;
    const selection = { id: null, request: 0, job: null, cancelling: false };
    current = selection;
    result = null;
    clearTimeout(timer);
    clearCancellation();
    $("classify").disabled = true;
    $("analysis-panel").hidden = false;
    $("analysis-title").textContent = "Analysis";
    $("analysis-identity").textContent = "";
    $("job-state").textContent = "";
    $("job-detail").textContent = "Uploading the experiment.";
    $("progress").hidden = false;
    $("result").hidden = true;
    retry(null);
    clearReport();
    notice();
    try {
      if (!snapshot)
        snapshot = await json("/api/v1/snapshots", { method: "POST", body: exportText });
      const analysis = await json("/api/v1/analyses", {
        method: "POST",
        body: { snapshot_id: snapshot.snapshot_id, model_version_id: "active" },
        headers: { "Idempotency-Key": intakeKey },
      });
      if (current === selection) await openAnalysis(analysis.analysis_id);
      await history(0);
    } catch (error) {
      if (current !== selection || workspace !== epoch) return;
      $("progress").hidden = true;
      $("job-detail").textContent = "The request was not confirmed. Retry to check or continue it.";
      retry(submit);
      notice(error.message, true);
    } finally {
      if (workspace === epoch) {
        busy = false;
        $("classify").disabled = !session || !exportText;
      }
    }
  }
  async function openAnalysis(id) {
    clearTimeout(timer);
    clearReport();
    clearCancellation();
    // Object identity also rejects an old response after selecting A, B, then A again.
    const selection = { id, request: 0, job: null, cancelling: false };
    current = selection;
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
      $("analysis-title").textContent = "Analysis";
      $("analysis-identity").textContent =
        `${id} · Model ${job.model?.model_version_id || "pending"}`;
      $("job-state").textContent =
        job.state === "succeeded" ? "Complete" : job.state.replaceAll("_", " ");
      const pending = ["queued", "running", "cancel_requested"].includes(job.state);
      $("progress").hidden = !pending;
      selection.job = pending && job.state !== "cancel_requested" ? job.operation_id : null;
      $("cancel").hidden = selection.job === null;
      $("cancel").disabled = selection.cancelling || selection.job === null;
      $("job-detail").textContent =
        job.failure?.message ||
        (job.state === "succeeded"
          ? ""
          : job.state === "cancelled"
            ? "This analysis was cancelled."
            : job.state === "cancel_requested"
              ? "Cancelling the analysis."
              : stages[job.stage] || "Processing the analysis.");
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
    $("analysis-identity").textContent += ` · Run ${data.source.run_id}`;
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
      return;
    }
    clearReport();
    $("download-report").href = `/api/v1/analyses/${id}/report`;
    $("download-report").download = `regimex-analysis-${id}.html`;
    $("download-report").hidden = false;
    $("report-preview").src = `/api/v1/analyses/${id}/report/preview`;
    $("report-preview").hidden = false;
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
