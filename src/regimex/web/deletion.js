import {$, notice} from "./ui.js";
import {bytes} from "./storage.js";

const PAGE_SIZE = 20;
const ROOT = "/api/v1/deletions";

function counts(value) {
  return [
    value.experiment_count ? `${value.experiment_count} ${value.experiment_count === 1 ? "experiment" : "experiments"}` : "",
    value.analysis_count ? `${value.analysis_count} ${value.analysis_count === 1 ? "analysis" : "analyses"}` : "",
  ].filter(Boolean).join(" and ");
}

export function createDeletion({session, onApplied}) {
  const dialog = $("deletion-dialog");
  let selection = null;
  let listRequest = 0;
  let offset = 0;
  let hasMore = false;

  function current(value) {
    return session.value && value === selection && value.epoch === session.epoch;
  }

  function close() {
    selection = null;
    if (dialog.open) dialog.close();
  }

  function reset() {
    close();
    listRequest++;
    offset = 0;
    hasMore = false;
    $("pending-deletions").hidden = true;
    $("pending-deletions").open = false;
    $("pending-deletion-list").replaceChildren();
    $("pending-deletion-status").textContent = "";
    $("deletion-summary").textContent = $("deletion-storage").textContent = "";
    $("deletion-blockers").replaceChildren();
    $("deletion-status").textContent = "";
  }

  function busy(value, state) {
    value.busy = state;
    $("deletion-confirm").disabled = state;
    $("deletion-refresh").disabled = state;
    $("deletion-cancel").disabled = state;
  }

  function start(target) {
    close();
    const value = {target, epoch: session.epoch, plan: null, receipt: null,
      busy: false, confirmed: false, uncertain: false};
    selection = value;
    $("deletion-title").textContent = target.kind === "experiment" ? "Delete experiment" : "Delete analysis";
    $("deletion-summary").textContent = target.title || target.id;
    $("deletion-storage").hidden = true;
    $("deletion-blockers").hidden = true;
    $("deletion-blockers").replaceChildren();
    $("deletion-warning").hidden = true;
    $("deletion-status").textContent = "Loading deletion preview…";
    $("deletion-confirm").textContent = "Delete";
    $("deletion-confirm").disabled = true;
    $("deletion-refresh").hidden = true;
    $("deletion-cancel").textContent = "Cancel";
    $("deletion-cancel").disabled = false;
    dialog.showModal();
    return value;
  }

  function renderPlan(value) {
    const plan = value.plan;
    $("deletion-summary").textContent = `${value.target.title || value.target.id} · ${counts(plan)}`;
    $("deletion-storage").hidden = false;
    $("deletion-storage").textContent = plan.size_bytes
      ? `${bytes(plan.size_bytes)} will be released after the stored files are removed.`
      : "There are no stored files to remove.";
    const list = $("deletion-blockers");
    list.replaceChildren(...plan.blockers.map((blocker) => {
      const item = document.createElement("li");
      item.textContent = blocker.message;
      return item;
    }));
    list.hidden = plan.blockers.length === 0;
    $("deletion-warning").hidden = plan.blockers.length > 0;
    $("deletion-status").textContent = plan.blockers.length ? "Resolve these dependencies before deleting." : "";
    $("deletion-confirm").textContent = "Delete";
    $("deletion-confirm").disabled = plan.blockers.length > 0;
    $("deletion-refresh").hidden = plan.blockers.length === 0;
  }

  async function preview(value) {
    if (!current(value) || value.busy || value.confirmed) return;
    value.plan = null;
    busy(value, true);
    $("deletion-status").textContent = "Loading deletion preview…";
    $("deletion-warning").hidden = true;
    try {
      const path = value.target.kind === "experiment" ? "experiments" : "analyses";
      const plan = await session.json(`/api/v1/${path}/${value.target.id}/deletion-preview`, {method: "POST"});
      if (!current(value)) return;
      value.plan = plan;
      busy(value, false);
      renderPlan(value);
    } catch (error) {
      if (!current(value)) return;
      busy(value, false);
      $("deletion-confirm").disabled = true;
      $("deletion-refresh").hidden = false;
      $("deletion-status").textContent = error.message;
    }
  }

  function received(value, receipt) {
    value.receipt = receipt;
    value.uncertain = false;
    busy(value, false);
    onApplied(value.target, receipt);
    void load();
    if (receipt.state === "completed") {
      close();
      notice(`Deleted ${counts(receipt)}.${receipt.reclaimed_bytes ? ` Released ${bytes(receipt.reclaimed_bytes)}.` : ""}`);
      return;
    }
    $("deletion-title").textContent = "Deletion needs cleanup";
    $("deletion-summary").textContent = `${counts(receipt)} removed from history.`;
    $("deletion-storage").hidden = false;
    $("deletion-storage").textContent = `${bytes(receipt.pending_bytes)} still counts toward your storage limit.`;
    $("deletion-blockers").hidden = true;
    $("deletion-warning").hidden = true;
    $("deletion-status").textContent = receipt.failure || "Some files could not be removed. Retry cleanup to finish.";
    $("deletion-confirm").textContent = "Retry cleanup";
    $("deletion-confirm").disabled = false;
    $("deletion-refresh").hidden = true;
    $("deletion-cancel").textContent = "Close";
  }

  async function check(value) {
    busy(value, true);
    $("deletion-status").textContent = "Checking deletion status…";
    try {
      const receipt = await session.json(`${ROOT}/${value.plan.plan_id}`);
      if (current(value)) received(value, receipt);
    } catch (error) {
      if (!current(value)) return;
      busy(value, false);
      if (error.status === 404) {
        value.confirmed = value.uncertain = false;
        $("deletion-confirm").disabled = true;
        $("deletion-refresh").hidden = false;
        $("deletion-status").textContent = "Deletion was not recorded. Refresh the preview before trying again.";
      } else {
        $("deletion-status").textContent = error.message;
        $("deletion-confirm").textContent = "Check deletion";
      }
    }
  }

  async function confirm() {
    const value = selection;
    if (!value || !current(value) || value.busy || !value.plan || $("deletion-confirm").disabled) return;
    if (value.uncertain) return check(value);
    if (!value.confirmed && value.plan.blockers?.length) return;
    if (!value.confirmed && Date.parse(value.plan.expires_at) <= Date.now()) {
      $("deletion-confirm").disabled = true;
      $("deletion-refresh").hidden = false;
      $("deletion-status").textContent = "This preview has expired. Refresh it before deleting.";
      return;
    }
    value.confirmed = true;
    busy(value, true);
    $("deletion-status").textContent = "Deleting…";
    try {
      const receipt = await session.json(`${ROOT}/${value.plan.plan_id}`, {
        method: "POST", body: {sha256: value.plan.sha256},
      });
      if (current(value)) received(value, receipt);
    } catch (error) {
      if (!current(value)) return;
      busy(value, false);
      value.uncertain = true;
      $("deletion-confirm").textContent = "Check deletion";
      $("deletion-refresh").hidden = true;
      $("deletion-cancel").textContent = "Close";
      $("deletion-status").textContent = `${error.message} Check deletion status before trying again.`;
      void load();
    }
  }

  async function open(kind, id, title = "") {
    if (!session.value) return;
    return preview(start({kind, id, title}));
  }

  async function resume(receipt) {
    if (!session.value) return;
    const value = start({kind: receipt.target_kind, id: receipt.target_id});
    value.plan = {plan_id: receipt.deletion_id, sha256: receipt.sha256};
    value.confirmed = value.uncertain = true;
    return check(value);
  }

  async function load(page = 0) {
    if (!session.value) return;
    const epoch = session.epoch;
    const request = ++listRequest;
    const active = () => session.value && epoch === session.epoch && request === listRequest;
    $("pending-deletions-previous").disabled = $("pending-deletions-next").disabled = true;
    try {
      const data = await session.json(`${ROOT}?limit=${PAGE_SIZE + 1}&offset=${page}`);
      if (!active()) return;
      offset = page;
      hasMore = data.items.length > PAGE_SIZE;
      const items = data.items.slice(0, PAGE_SIZE);
      $("pending-deletions").hidden = !items.length && page === 0;
      $("pending-deletion-status").textContent = items.length ? "" : "No pending deletions on this page.";
      $("pending-deletion-list").replaceChildren(...items.map((receipt) => {
        const row = document.createElement("li");
        const button = document.createElement("button");
        button.type = "button";
        button.className = "quiet";
        button.textContent = `${receipt.target_kind === "experiment" ? "Experiment" : "Analysis"} ${receipt.target_id.slice(-8)} · ${bytes(receipt.pending_bytes)} pending`;
        button.addEventListener("click", () => {
          if (active()) return resume(receipt);
        });
        row.append(button);
        return row;
      }));
    } catch (error) {
      if (active()) {
        $("pending-deletions").hidden = false;
        $("pending-deletion-status").textContent = error.message;
      }
    } finally {
      if (active()) {
        $("pending-deletions-previous").hidden = offset === 0;
        $("pending-deletions-next").hidden = !hasMore;
        $("pending-deletions-previous").disabled = offset === 0;
        $("pending-deletions-next").disabled = !hasMore;
      }
    }
  }

  $("deletion-confirm").addEventListener("click", confirm);
  $("deletion-refresh").addEventListener("click", () => selection && preview(selection));
  $("deletion-cancel").addEventListener("click", () => { if (!selection?.busy) close(); });
  dialog.addEventListener("cancel", (event) => {
    event.preventDefault();
    if (!selection?.busy) close();
  });
  $("pending-deletions-refresh").addEventListener("click", () => { void load(0); });
  $("storage-refresh").addEventListener("click", () => { void load(0); });
  $("pending-deletions-previous").addEventListener("click", () => { void load(Math.max(0, offset - PAGE_SIZE)); });
  $("pending-deletions-next").addEventListener("click", () => { void load(offset + PAGE_SIZE); });
  return {open, close, reset, load};
}
