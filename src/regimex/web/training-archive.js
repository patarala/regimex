import {$} from "./ui.js";

const selectionLimit = 100;
const bulkActions = [["training-archive-selected", true], ["training-restore-selected", false]];
export const finished = job => ["succeeded", "failed", "cancelled"].includes(job.state);

export function createRunArchive({apply, renderRows}) {
  const selected = new Map();
  const toggle = $("training-select-runs");
  const detail = $("training-archive-run");
  let selecting = false, saving = false, current = null;

  function controls() {
    toggle.textContent = selecting ? "Done" : "Select runs";
    toggle.setAttribute("aria-pressed", String(selecting));
    toggle.disabled = saving;
    $("training-selection").hidden = !selecting;
    $("training-selection-count").textContent = selected.size === selectionLimit
      ? selected.size + " selected (limit reached)" : selected.size + " selected";
    const statuses = [...selected.values()];
    for (const [id, archived] of bulkActions) {
      const button = $(id);
      const count = statuses.filter(alreadyArchived => alreadyArchived !== archived).length;
      button.hidden = !count;
      button.disabled = saving;
      const verb = archived ? "Archive" : "Restore";
      button.textContent = count === selected.size
        ? verb + " selected" : verb + " " + count + (count === 1 ? " run" : " runs");
    }
    detail.hidden = !current || !finished(current.job);
    detail.disabled = saving;
    detail.textContent = current?.archived_at ? "Restore run" : "Archive run";
  }

  async function save(ids, archived) {
    if (saving || !ids.length) return;
    const trigger = document.activeElement;
    saving = true;
    renderRows();
    try {
      if (await apply(ids, archived)) {
        for (const id of ids) selected.delete(id);
        if (!selected.size) selecting = false;
      }
    } finally {
      saving = false;
      renderRows();
      if (bulkActions.some(([id]) => $(id) === trigger) && document.activeElement === document.body &&
          !$("training").hidden) {
        (toggle.hidden ? $("training-history-view") : toggle).focus({preventScroll: true});
      }
    }
  }

  for (const [id, archived] of bulkActions) {
    $(id).addEventListener("click", () => {
      const ids = [...selected].filter(([, state]) => state !== archived).map(([id]) => id);
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
    input.disabled = saving || !finished(row.job) || (!input.checked && selected.size >= selectionLimit);
    input.addEventListener("change", () => {
      if (input.checked) selected.set(row.run_id, Boolean(row.archived_at));
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
    showRun(row) { current = row; controls(); },
    showPage(rows) {
      for (const row of rows) {
        if (selected.has(row.run_id)) selected.set(row.run_id, Boolean(row.archived_at));
      }
      toggle.hidden = !selecting && !rows.some(row => finished(row.job));
      controls();
    },
    reset() { selecting = false; selected.clear(); toggle.hidden = true; controls(); },
  };
}
