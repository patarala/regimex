import {notice} from "./ui.js";

const PAGE_SIZE = 20;
const stateLabels = {
  queued: "Queued", running: "Running", cancel_requested: "Cancelling",
  cancelled: "Cancelled", succeeded: "Complete", failed: "Failed",
};

function element(tag, className = "", text = "") {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = text;
  return node;
}

function button(className, text) {
  const node = element("button", className, text);
  node.type = "button";
  return node;
}

function shortDate(value) {
  return new Date(value).toLocaleString(undefined, {dateStyle: "medium", timeStyle: "short"});
}

export function createExperimentHistory({session, experiment, filters, isActive, getSelection,
  openAnalysis, compareAnalyses, deleteExperiment, refresh}) {
  let value = experiment;
  let requestNumber = 0;
  let offset = 0;
  let hasMore = false;
  let items = [];
  let loading = false;
  let pendingOffset = null;
  let loaded = false;
  let editRevision = null;
  let editing = 0;
  const selected = new Map();
  const node = element("li", "history-experiment");
  node.dataset.snapshotId = value.snapshot_id;
  const latest = button("history-entry experiment-open", "");
  const name = element("strong");
  name.setAttribute("dir", "auto");
  const model = element("span");
  const instruments = element("span", "history-instruments");
  const date = element("time");
  const state = element("span", "history-state");
  latest.append(name, model, instruments, date, state);
  latest.addEventListener("click", () => {
    if (isActive() && value.latest_analysis) return openAnalysis(value.latest_analysis.analysis_id);
  });

  const tools = element("div", "experiment-tools");
  const expand = button("quiet experiment-expand", "");
  const rename = button("quiet experiment-rename", "Rename");
  const remove = button("quiet delete-action experiment-delete", "Delete experiment");
  remove.addEventListener("click", () => {
    if (isActive()) return deleteExperiment(value.snapshot_id, title());
  });
  tools.append(expand, rename, remove);
  const editor = element("form", "experiment-name");
  editor.hidden = true;
  const label = element("label", "", "Experiment name");
  const input = element("input");
  input.type = "text";
  input.setAttribute("dir", "auto");
  input.placeholder = "Use instrument names";
  label.append(input);
  const save = button("quiet", "Save");
  save.type = "submit";
  const cancel = button("quiet", "Cancel");
  const nameActions = element("div", "experiment-tools");
  nameActions.append(save, cancel);
  const nameStatus = element("p", "hint");
  nameStatus.setAttribute("role", "status");
  editor.append(label, nameActions, nameStatus);

  const runs = element("div", "experiment-runs");
  runs.hidden = true;
  runs.id = `experiment-runs-${value.snapshot_id}`;
  expand.setAttribute("aria-controls", runs.id);
  expand.setAttribute("aria-expanded", "false");
  const list = element("ul", "experiment-run-list");
  const runStatus = element("p", "hint");
  runStatus.setAttribute("role", "status");
  const pages = element("div", "experiment-tools experiment-run-pages");
  const previous = button("quiet", "Newer");
  const next = button("quiet", "Earlier");
  const reload = button("quiet", "Refresh");
  pages.append(previous, next, reload);
  const compare = button("quiet experiment-compare", "Compare selected");
  compare.disabled = true;
  const chosen = element("p", "hint comparison-selection");
  runs.append(compare, chosen, list, runStatus, pages);
  node.append(latest, tools, editor, runs);

  function title() {
    return value.display_name || value.instrument_ids.join(", ") || "Market experiment";
  }
  function update(nextValue) {
    // A list request started before a rename must not restore its older name.
    if (nextValue.name_revision < value.name_revision)
      nextValue = {...nextValue, display_name: value.display_name, name_revision: value.name_revision};
    value = nextValue;
    const analysis = value.latest_analysis;
    name.textContent = title();
    instruments.textContent = value.display_name ? value.instrument_ids.join(", ") : "";
    model.textContent = analysis ? analysis.job.model?.model_version_id || "Model unavailable" : "No analyses";
    date.dateTime = analysis?.job.created_at || value.created_at;
    date.textContent = shortDate(date.dateTime);
    state.dataset.state = analysis?.job.state || "saved";
    state.textContent = analysis ? stateLabels[analysis.job.state] || analysis.job.state : "Saved";
    latest.dataset.analysisId = analysis?.analysis_id || "";
    latest.disabled = !analysis;
    latest.title = analysis ? `Open latest analysis · ${value.run_id}` : value.run_id;
    latest.setAttribute("aria-label", analysis ? `Open latest analysis of ${title()}` : title());
    const count = value.matching_analysis_count === value.analysis_count
      ? value.analysis_count : `${value.matching_analysis_count} of ${value.analysis_count}`;
    expand.textContent = `${runs.hidden ? "Show" : "Hide"} ${count} ${value.analysis_count === 1 ? "analysis" : "analyses"}`;
    expand.hidden = value.analysis_count === 0;
    if (!analysis) runs.hidden = true;
    markSelected();
  }
  function markSelected() {
    for (const entry of [latest, ...list.children].map((item) => item === latest ? item : item.children[1])) {
      if (entry.dataset.analysisId === getSelection()?.id) entry.setAttribute("aria-current", "true");
      else entry.removeAttribute("aria-current");
    }
  }
  function updateComparison() {
    compare.disabled = selected.size !== 2;
    chosen.textContent = selected.size
      ? `${selected.size} selected · ${[...selected.values()].map((item) =>
        `${item.job.model?.model_version_id || "Model unavailable"} (${shortDate(item.job.created_at)})`).join(", ")}`
      : "Select two completed analyses to compare.";
    for (const row of list.children) {
      const checkbox = row.children[0];
      checkbox.checked = selected.has(checkbox.dataset.analysisId);
      checkbox.disabled = checkbox.dataset.available !== "true"
        || (selected.size === 2 && !checkbox.checked);
    }
  }
  function renderRuns() {
    list.replaceChildren();
    items.forEach((item) => {
      const row = element("li", "experiment-run");
      const checkbox = element("input");
      checkbox.type = "checkbox";
      checkbox.dataset.analysisId = item.analysis_id;
      checkbox.dataset.available = String(item.job.state === "succeeded" && item.result_available);
      checkbox.setAttribute("aria-label", `Select ${item.job.model?.model_version_id || "analysis"}, ${shortDate(item.job.created_at)} for comparison`);
      checkbox.addEventListener("change", () => {
        if (!isActive()) return;
        if (checkbox.checked && checkbox.dataset.available === "true" && selected.size < 2)
          selected.set(item.analysis_id, item);
        else selected.delete(item.analysis_id);
        updateComparison();
      });
      const entry = button("history-entry", "");
      entry.dataset.analysisId = item.analysis_id;
      entry.append(element("strong", "", item.job.model?.model_version_id || "Model unavailable"),
        element("time", "", shortDate(item.job.created_at)),
        element("span", "history-state", stateLabels[item.job.state] || item.job.state));
      entry.addEventListener("click", () => openAnalysis(item.analysis_id));
      row.append(checkbox, entry);
      list.append(row);
    });
    updateComparison();
    markSelected();
  }
  async function load(page = offset) {
    const request = ++requestNumber;
    const workspace = session.epoch;
    const active = () => isActive() && workspace === session.epoch && request === requestNumber;
    loading = true;
    pendingOffset = page;
    previous.disabled = next.disabled = reload.disabled = true;
    runStatus.textContent = "Loading analyses…";
    try {
      const query = new URLSearchParams({snapshot_id: value.snapshot_id, limit: PAGE_SIZE + 1, offset: page});
      if (filters.q) query.set("q", filters.q);
      if (filters.state) query.set("state", filters.state);
      const data = await session.json(`/api/v1/analyses?${query}`);
      if (!active()) return;
      offset = page;
      hasMore = data.items.length > PAGE_SIZE;
      items = data.items.slice(0, PAGE_SIZE).filter((item) => item.snapshot_id === value.snapshot_id);
      for (const item of items) {
        if (selected.has(item.analysis_id)) {
          if (item.job.state === "succeeded" && item.result_available) selected.set(item.analysis_id, item);
          else selected.delete(item.analysis_id);
        }
      }
      loaded = true;
      renderRuns();
      runStatus.textContent = items.length ? "" : "No analyses on this page.";
    } catch (error) {
      if (active()) runStatus.textContent = error.message;
    } finally {
      if (active()) {
        loading = false;
        pendingOffset = null;
        previous.hidden = offset === 0;
        next.hidden = !hasMore;
        previous.disabled = offset === 0;
        next.disabled = !hasMore;
        reload.disabled = false;
      }
    }
  }
  function observe(analysis) {
    if (value.latest_analysis?.analysis_id === analysis.analysis_id)
      update({...value, latest_analysis: {...value.latest_analysis, ...analysis}});
    const index = items.findIndex((item) => item.analysis_id === analysis.analysis_id);
    if (index >= 0 && items[index].job.state !== analysis.job.state) {
      items[index] = {...items[index], ...analysis};
      if (analysis.job.state !== "succeeded") selected.delete(analysis.analysis_id);
      renderRuns();
    }
  }
  expand.addEventListener("click", () => {
    if (!isActive()) return;
    runs.hidden = !runs.hidden;
    expand.setAttribute("aria-expanded", String(!runs.hidden));
    update(value);
    if (!runs.hidden && !loaded) return load(0);
  });
  previous.addEventListener("click", () => load(Math.max(0, offset - PAGE_SIZE)));
  next.addEventListener("click", () => load(offset + PAGE_SIZE));
  reload.addEventListener("click", () => load(0));
  compare.addEventListener("click", () => {
    if (!isActive() || selected.size !== 2) return;
    const workspace = session.epoch;
    return compareAnalyses(...selected.values()).catch((error) => {
      if (isActive() && workspace === session.epoch) notice(error.message, true);
    });
  });
  rename.addEventListener("click", () => {
    if (!isActive()) return;
    editing++;
    editRevision = value.name_revision;
    input.value = value.display_name || "";
    input.disabled = save.disabled = false;
    nameStatus.textContent = "";
    editor.hidden = false;
    input.focus();
  });
  cancel.addEventListener("click", () => {
    editing++;
    editor.hidden = true;
  });
  editor.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (save.disabled || !isActive()) return;
    const displayName = input.value.trim().normalize("NFC");
    if (Array.from(displayName).length > 120) {
      nameStatus.textContent = "Use at most 120 characters for the experiment name.";
      return;
    }
    const edit = editing;
    const workspace = session.epoch;
    const active = () => isActive() && workspace === session.epoch && edit === editing;
    const body = {display_name: displayName || null, expected_revision: editRevision};
    input.disabled = save.disabled = true;
    nameStatus.textContent = "Saving…";
    try {
      const saved = await session.json(`/api/v1/experiments/${value.snapshot_id}/name`, {method: "PATCH", body});
      if (!active()) return;
      update({...value, display_name: saved.display_name, name_revision: saved.revision});
      editor.hidden = true;
      await refresh().catch((error) => {
        if (active()) notice(error.message, true);
      });
    } catch (error) {
      if (!active()) return;
      nameStatus.textContent = error.message;
      if (error.status === 409) {
        try {
          await refresh();
          if (active()) {
            editRevision = value.name_revision;
            nameStatus.textContent = "The name changed elsewhere. Review the current name above and save again.";
          }
        } catch (refreshError) {
          if (active()) nameStatus.textContent = refreshError.message;
        }
      }
    } finally {
      if (active()) input.disabled = save.disabled = false;
    }
  });
  update(value);
  return {node, update, observe, markSelected, get loading() { return loading; },
    refreshRuns: () => !runs.hidden && (loaded || loading) ? load(pendingOffset ?? offset) : Promise.resolve()};
}
