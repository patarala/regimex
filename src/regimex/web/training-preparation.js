import {$, element} from "./ui.js";
import {isTransientRequestError} from "./session.js";

const api = "/api/v1/training-preparations";
const options = "/api/v1/training-options/";
const inventoryStatement = "I confirm that all known prior uses of data from these sources are registered.";
const labels = {
  development: "Generate development experiments", final: "Generate final experiments",
  train: "Label training experiments", validation: "Label validation experiments", test: "Seal final test labels",
};
const pageSize = 10;

export function createDataPreparation({session, navigate, onUse}) {
  const panel = $("data-preparation"), form = $("data-preparation-form");
  const action = $("preparation-action"), status = $("preparation-status"), check = $("preparation-attest");
  const recent = $("preparation-recent");
  const legacy = $("preparation-resume");
  const sizes = [$("preparation-development"), $("preparation-final")];
  const defaults = sizes.map(input => input.valueAsNumber);
  const schemaLimit = Number(sizes[0].max);
  let id, record, inventory, pending, olderSetup, timer, busy = false, limitReady = false;
  let revision = 0, historyRevision = 0, historyOffset = 0, retryDelay = 1500;
  const active = stamp => stamp === revision && session.value?.principal.role === "admin";

  function fields() {
    return Object.fromEntries(["development", "final", "steps"].map(name =>
      [name, $("preparation-" + name).valueAsNumber]));
  }
  function counts() {
    const {development, final} = fields();
    const train = Math.min(development - 2, Math.max(1, Math.floor(development * 0.7)));
    const validation = Math.min(development - train - 1, Math.max(1, Math.floor(development * 0.25)));
    $("preparation-counts").textContent = development >= 3 && final >= 3
      ? `${train} training and ${validation} validation experiments. ${final - 2} separate experiments are reserved for final testing.`
      : "";
  }
  function stageLabel(row) {
    if (row.selection_closed) return "Selection closed";
    if (row.benchmark_id) return "Benchmark reserved";
    if (row.outputs.test) return "Ready for review";
    return labels[row.stage] + " · " + row.state.replaceAll("_", " ");
  }
  function show(message = "", error = false) {
    const ready = Boolean(record?.benchmark_id) && !record.selection_closed;
    const failed = ["failed", "cancelled"].includes(record?.state);
    const review = Boolean(inventory) && !ready;
    $("preparation-size").disabled = Boolean(id) || busy;
    $("preparation-review").hidden = !review;
    $("preparation-inventory").textContent = review
      ? `The service-wide inventory has ${inventory.dataset_count} registered datasets. Confirm its full scope to register the development dataset and reserve the final test partition.` : "";
    $("preparation-source-count").textContent = review ? `Review ${inventory.source_namespaces.length} source namespaces` : "";
    $("preparation-datasets").textContent = review ? `Development: ${record.outputs.development} · Final: ${record.outputs.final}` : "";
    $("preparation-sources").textContent = review ? inventory.source_namespaces.join("\n") : "";
    check.disabled = busy;
    action.hidden = Boolean(record && !failed && !review && !ready && !error);
    action.disabled = busy || (!id && !limitReady) || (review && !check.checked);
    action.textContent = ready ? "Use prepared data" : review ? "Reserve benchmark" : failed ? "Retry stage"
      : id ? "Refresh preparation" : pending ? "Retry request" : "Generate datasets";
    $("preparation-new").hidden = !id;
    legacy.hidden = Boolean(id) || !olderSetup;
    legacy.disabled = busy;
    status.textContent = message || (record ? [stageLabel(record), record.failure].filter(Boolean).join(". ") : "");
    status.hidden = !status.textContent;
    status.classList.toggle("error", error || failed);
  }

  async function refresh(stamp = revision) {
    clearTimeout(timer);
    try {
      const next = await session.json(api + "/" + id);
      if (!active(stamp)) return;
      record = next;
      for (const field of ["development", "final", "steps"]) $("preparation-" + field).value = record.request[field];
      counts();
      if (record.outputs.test && !record.benchmark_id) {
        const query = new URLSearchParams({dataset_id: record.outputs.development, final_dataset_id: record.outputs.final});
        const reviewed = await session.json(options + "inventory?" + query);
        if (!active(stamp)) return;
        if (inventory?.revision !== reviewed.revision) check.checked = false;
        inventory = reviewed;
      }
      busy = false;
      show();
      retryDelay = 1500;
      if (!document.hidden && !record.outputs.test && !["failed", "cancelled"].includes(record.state))
        timer = setTimeout(() => void refresh(stamp), 1500);
    } catch (error) {
      if (!active(stamp)) return;
      busy = false;
      if (isTransientRequestError(error)) {
        show("Could not read preparation progress. Reconnecting…");
        timer = setTimeout(() => void refresh(stamp), retryDelay);
        retryDelay = Math.min(30000, retryDelay * 2);
      } else show(error.message, true);
    }
  }
  function close() {
    ++revision;
    ++historyRevision;
    clearTimeout(timer);
    record = id = inventory = pending = olderSetup = null;
    busy = false;
    check.checked = false;
    recent.open = panel.open = false;
    form.reset();
    show();
  }
  function select(idToShow) {
    close();
    id = idToShow;
    if (!id && session.value?.principal.role === "admin") {
      try {
        const key = "regimex.training-preparation.v1." + session.value.principal.owner_id;
        const saved = JSON.parse(localStorage.getItem(key));
        if (saved?.id) olderSetup = {key, saved};
      } catch { /* Storage can be unavailable in private sessions. */ }
    }
    show();
    counts();
    if (id) {
      busy = true;
      panel.open = true;
      show("Loading preparation…");
      void refresh();
    }
  }
  legacy.addEventListener("click", async () => {
    if (!olderSetup || busy) return;
    const {key, saved} = olderSetup, stamp = ++revision;
    busy = true;
    show("Restoring the saved preparation…");
    try {
      const accepted = await session.json(api + "/" + encodeURIComponent(saved.id), {
        method: "PUT", body: {development: saved.development, final: saved.final, steps: saved.steps,
          browser_seeds: {development: saved.devSeed, final: saved.finalSeed}},
      });
      if (!active(stamp)) return;
      try { localStorage.removeItem(key); } catch { /* Replaying is idempotent. */ }
      await navigate(accepted.workspace_url);
    } catch (error) {
      if (active(stamp)) { busy = false; show(error.message, true); }
    }
  });
  async function submit(event) {
    event.preventDefault();
    if (busy || (!id && !limitReady) || session.value?.principal.role !== "admin") return;
    clearTimeout(timer);
    const stamp = ++revision;
    busy = true;
    show();
    try {
      if (!id) {
        const request = fields();
        if (!pending || JSON.stringify(pending.request) !== JSON.stringify(request))
          pending = {id: crypto.randomUUID(), request};
        const accepted = await session.json(api + "/" + pending.id, {method: "PUT", body: request});
        if (active(stamp)) await navigate(accepted.workspace_url);
      } else if (record?.benchmark_id && !record.selection_closed) {
        const used = await onUse(record.outputs, record.benchmark_id);
        if (!active(stamp)) return;
        if (used) {
          await navigate("/training/new");
          $("training-dataset").focus();
        } else { busy = false; show("The form changed. Select the prepared data again."); }
      } else if (inventory && check.checked) {
        await session.json(options + "selection", {method: "POST", body: {
          dataset_id: record.outputs.development, final_dataset_id: record.outputs.final,
          final_reference_set_id: record.outputs.test, expected_registry_revision: inventory.revision,
          inventory_statement: inventoryStatement,
        }});
        if (active(stamp)) await refresh(stamp);
      } else if (["failed", "cancelled"].includes(record?.state)) {
        await session.json(api + "/" + id + "/retry", {method: "POST", body: {operation_id: record.operation_id}});
        if (active(stamp)) await refresh(stamp);
      } else await refresh(stamp);
    } catch (error) {
      if (!active(stamp)) return;
      busy = false;
      inventory = null;
      check.checked = false;
      show(error.message, true);
    }
  }
  async function list(offset) {
    const stamp = ++historyRevision;
    const rows = $("preparation-list");
    const previous = $("preparation-previous"), next = $("preparation-next");
    previous.disabled = next.disabled = true;
    try {
      const page = await session.json(api + `?limit=${pageSize + 1}&offset=${offset}`);
      if (stamp !== historyRevision) return;
      historyOffset = offset;
      rows.replaceChildren(...page.items.slice(0, pageSize).map(row => {
        const item = element("li");
        const link = element("a", `${new Date(row.created_at).toLocaleString()} · ${stageLabel(row)}`);
        link.href = row.workspace_url;
        link.dataset.workspaceLink = "";
        item.append(link);
        return item;
      }));
      if (!page.items.length) rows.append(element("li", "No saved preparations."));
      previous.hidden = !offset;
      next.hidden = page.items.length <= pageSize;
    } catch (error) {
      if (stamp === historyRevision) rows.replaceChildren(element("li", error.message));
    } finally {
      if (stamp === historyRevision) previous.disabled = next.disabled = false;
    }
  }
  document.addEventListener("visibilitychange", () => {
    clearTimeout(timer);
    if (!document.hidden && id && !busy) void refresh(++revision);
  });
  recent.addEventListener("toggle", () => { if (recent.open) void list(0); });
  $("preparation-previous").addEventListener("click", () => void list(Math.max(0, historyOffset - pageSize)));
  $("preparation-next").addEventListener("click", () => void list(historyOffset + pageSize));
  form.addEventListener("submit", submit);
  $("preparation-size").addEventListener("input", counts);
  check.addEventListener("change", () => show());
  $("preparation-new").addEventListener("click", async event => {
    if (event.button || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    await navigate("/training/new");
    panel.open = true;
    sizes[0].focus();
  });
  function setExperimentLimit(limit = schemaLimit) {
    sizes.forEach((input, index) => {
      input.max = String(limit);
      input.defaultValue = String(Math.min(defaults[index], limit));
    });
    limitReady = true;
    if (!id) { counts(); show(); }
  }
  function open(idToShow) { limitReady = false; select(idToShow); }
  return {open, select, close, setExperimentLimit};
}
