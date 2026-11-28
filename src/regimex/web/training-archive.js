import {$} from "./ui.js";

const selectionLimit = 100;
const bulkActions = [["training-archive-selected", true], ["training-restore-selected", false]];
export const finished = job => ["succeeded", "failed", "cancelled"].includes(job.state);
const archiveState = row => finished(row.job) ? Boolean(row.archived_at) : null;

export function createRunArchive({session, apply, compare, renderRows}) {
  const selected = new Map();
  const toggle = $("training-select-runs");
  const detail = $("training-archive-run");
  let selecting = false, stale = false, current = null;
  let savingEpoch = null;
  const saving = () => savingEpoch === session.epoch;

  function controls() {
    toggle.textContent = selecting ? "Done" : "Select runs";
    toggle.setAttribute("aria-pressed", String(selecting));
    toggle.disabled = saving() || stale;
    $("training-selection").hidden = !selecting;
    $("training-compare-selected").disabled = selected.size !== 2 || saving() || stale;
    $("training-selection-count").textContent = selected.size === selectionLimit
      ? selectionLimit + " of " + selectionLimit + " selected" : selected.size + " selected";
    const view = $("training-history-view").value;
    const statuses = [...selected.values()];
    for (const [id, archived] of bulkActions) {
      const button = $(id);
      const count = statuses.filter(state => state === !archived).length;
      button.hidden = view !== "all" && view !== (archived ? "active" : "archived");
      button.disabled = !count || saving() || stale;
      if (count && count < selected.size) {
        button.title = count + (count === 1 ? " run" : " runs");
        button.setAttribute("aria-label", button.textContent + " (" + button.title + ")");
      } else {
        button.removeAttribute("title");
        button.removeAttribute("aria-label");
      }
    }
  }

  function runControl() {
    detail.hidden = !current || !finished(current.job);
    detail.disabled = saving();
    detail.textContent = current?.archived_at ? "Restore run" : "Archive run";
  }

  async function save(ids, archived) {
    if (saving() || !ids.length) return;
    const trigger = document.activeElement;
    const workspace = session.epoch;
    savingEpoch = workspace;
    runControl();
    renderRows();
    try {
      if (await apply(ids, archived) && workspace === session.epoch) {
        for (const id of ids) selected.delete(id);
        if (!selected.size) selecting = false;
      }
    } finally {
      if (workspace === session.epoch) {
        savingEpoch = null;
        runControl();
        renderRows();
        if (bulkActions.some(([id]) => $(id) === trigger) && document.activeElement === document.body &&
            !$("training").hidden) {
          (toggle.hidden ? $("training-history-view") : toggle).focus({preventScroll: true});
        }
      }
    }
  }

  $("training-compare-selected").addEventListener("click", () => {
    if (selecting && selected.size === 2 && !saving() && !stale) compare([...selected.keys()]);
  });
  for (const [id, archived] of bulkActions) {
    const button = $(id);
    button.addEventListener("click", () => {
      if (!selecting || button.hidden || button.disabled) return;
      const ids = [...selected].filter(([, state]) => state === !archived).map(([id]) => id);
      void save(ids, archived);
    });
  }
  detail.addEventListener("click", () => {
    if (current && finished(current.job)) void save([current.run_id], !current.archived_at);
  });
  toggle.addEventListener("click", () => {
    selecting = !selecting;
    selected.clear();
    renderRows();
  });

  function checkbox(row, label) {
    if (!selecting) return null;
    const input = document.createElement("input");
    input.type = "checkbox";
    input.dataset.selectRun = row.run_id;
    input.setAttribute("aria-label", "Select " + label);
    input.checked = selected.has(row.run_id);
    input.disabled = saving() || (!input.checked && selected.size >= selectionLimit);
    input.addEventListener("change", () => {
      if (input.disabled || saving() || stale || !selecting) return;
      if (input.checked) selected.set(row.run_id, archiveState(row));
      else selected.delete(row.run_id);
      renderRows();
    });
    const target = document.createElement("label");
    target.className = "training-run-select";
    target.append(input);
    return target;
  }
  return {
    checkbox,
    get selecting() { return selecting; },
    showRun(row) { current = row; runControl(); },
    showPage(rows, outdated = false) {
      stale = outdated;
      for (const row of rows) {
        if (selected.has(row.run_id)) selected.set(row.run_id, archiveState(row));
      }
      toggle.hidden = !selecting && !rows.length;
      controls();
    },
    reset() { selecting = false; selected.clear(); toggle.hidden = true; controls(); },
  };
}
