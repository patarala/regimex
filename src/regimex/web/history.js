import {$, notice} from "./ui.js";

const HISTORY_PAGE_SIZE = 20;
const stateLabels = {
  queued: "Queued", running: "Running", cancel_requested: "Cancelling",
  cancelled: "Cancelled", succeeded: "Complete", failed: "Failed",
};

export function createHistory({session, getSelection, openAnalysis}) {
  let historyGeneration = 0;
  let historyOffset = 0;
  let historyHasMore = false;
  let historyQuery = "";
  let historyState = "";
  let historyItems = [];
  let historyPendingOffset = null;
  function reset() {
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
    $("history-list").replaceChildren();
  }

  async function history(offset = historyOffset) {
    const generation = ++historyGeneration;
    const workspace = session.epoch;
    historyPendingOffset = offset;
    $("history-previous").disabled = $("history-next").disabled = true;
    $("history-loading").hidden = false;
    $("history-list").setAttribute("aria-busy", "true");
    try {
      // One lookahead item determines whether another bounded page exists.
      const query = new URLSearchParams({limit: HISTORY_PAGE_SIZE + 1, offset});
      if (historyQuery) query.set("q", historyQuery);
      if (historyState) query.set("state", historyState);
      const data = await session.json(`/api/v1/analyses?${query}`);
      if (generation !== historyGeneration || workspace !== session.epoch || !session.value) return;
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
      if (generation === historyGeneration && workspace === session.epoch) throw error;
    } finally {
      if (generation === historyGeneration && workspace === session.epoch) {
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
      if (button.dataset.analysisId === getSelection()?.id) button.setAttribute("aria-current", "true");
      else button.removeAttribute("aria-current");
    }
  }
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
  return {reset, load: history, update: updateHistory, markSelected: markSelectedHistory};
}
