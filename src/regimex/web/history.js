import {$, notice} from "./ui.js";
import {createExperimentHistory} from "./experiment-history.js";

const HISTORY_PAGE_SIZE = 20;
export function createHistory({session, getSelection, openAnalysis, compareAnalyses, deleteExperiment}) {
  let historyGeneration = 0;
  let historyOffset = 0;
  let historyHasMore = false;
  let historyQuery = "";
  let historyState = "";
  let historyItems = [];
  let historyPendingOffset = null;
  let records = new Map();
  function reset() {
    historyGeneration++;
    historyOffset = 0;
    historyHasMore = false;
    historyQuery = historyState = "";
    historyItems = [];
    records = new Map();
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
      const data = await session.json(`/api/v1/experiments?${query}`);
      if (generation !== historyGeneration || workspace !== session.epoch || !session.value) return;
      const items = data.items.slice(0, HISTORY_PAGE_SIZE);
      historyOffset = offset;
      historyHasMore = data.items.length > HISTORY_PAGE_SIZE;
      $("history-pages").hidden = offset === 0 && !historyHasMore;
      $("history-empty").hidden = items.length > 0;
      $("history-empty").textContent = offset > 0
        ? "No experiments on this page. Return to the previous page or refresh."
        : historyQuery || historyState ? "No experiments match these filters." : "No experiments yet.";
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
    const focused = document.activeElement;
    const scrollTop = $("history-list").scrollTop;
    const next = new Map();
    for (const item of historyItems) {
      let record = records.get(item.snapshot_id);
      if (record) record.update(item);
      else {
        record = createExperimentHistory({session, experiment: item,
          filters: {q: historyQuery, state: historyState}, getSelection, openAnalysis, compareAnalyses, deleteExperiment,
          isActive: () => records.get(item.snapshot_id) === record,
          refresh: () => history(historyOffset),
        });
      }
      next.set(item.snapshot_id, record);
    }
    records = next;
    $("history-list").replaceChildren(...[...records.values()].map((record) => record.node));
    $("history-list").scrollTop = scrollTop;
    if (focused && [...records.values()].some((record) => record.node.contains(focused))) focused.focus();
  }
  function applyHistoryFilters(event) {
    event?.preventDefault();
    historyQuery = $("history-search").value.trim();
    historyState = $("history-state").value;
    historyOffset = 0;
    historyHasMore = false;
    historyItems = [];
    records = new Map();
    renderHistory();
    $("history-empty").hidden = true;
    $("history-pages").hidden = true;
    $("history-clear").hidden = !historyQuery && !historyState;
    notice();
    return history(0).catch((error) => notice(error.message, true));
  }

  function updateHistory(value, changed) {
    for (const [snapshotId, record] of records) {
      record.observe(value);
      if (changed && snapshotId === value.snapshot_id) void record.refreshRuns();
    }
    // Refetch after a state transition so an older list response cannot undo it.
    if (changed && (historyState || historyPendingOffset !== null))
      history(historyPendingOffset ?? historyOffset).catch((error) => notice(error.message, true));
  }
  function markSelectedHistory() {
    for (const record of records.values()) record.markSelected();
  }
  function refreshAfterDeletion() {
    // Drop expanded lists and comparison selections before fetching the new history.
    records = new Map();
    historyItems = [];
    renderHistory();
    return history(0);
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
  return {reset, load: history, update: updateHistory, markSelected: markSelectedHistory,
    refreshAfterDeletion};
}
