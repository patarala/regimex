import {$} from "./ui.js";
import {isTransientRequestError} from "./session.js";

const api = "/api/v1/";
const inventoryStatement = "I confirm that all known prior uses of data from these sources are registered.";
const stages = [
  ["development", "Generate development experiments", "dataset_id"],
  ["final", "Generate final experiments", "dataset_id"],
  ["train", "Label training experiments", "reference_set_id"],
  ["validation", "Label validation experiments", "reference_set_id"],
  ["test", "Seal final test labels", "reference_set_id"],
];

function generationBody(plan, final) {
  return {
    mode: "generate",
    plan: {seed: final ? plan.finalSeed : plan.devSeed, step_count: plan.steps,
      experiment_count: final ? plan.final : plan.development},
    splits: {seed: 1729, train_fraction: final ? 0.001 : 0.7, validation_fraction: final ? 0.001 : 0.25},
  };
}

export function createDataPreparation({session, onUse}) {
  const panel = $("data-preparation");
  const form = $("data-preparation-form");
  const action = $("preparation-action");
  const status = $("preparation-status");
  const check = $("preparation-attest");
  let plan = null, key = "", revision = 0, timer, busy = false, inventory = null;
  let retryDelay = 1500;
  const drafts = new Map();
  function fields() {
    return Object.fromEntries(["development", "final", "steps"].map(id =>
      [id, $("preparation-" + id).valueAsNumber]));
  }
  function counts() {
    const {development, final} = fields();
    const train = Math.min(development - 2, Math.max(1, Math.floor(development * 0.7)));
    const validation = Math.min(development - train - 1, Math.max(1, Math.floor(development * 0.25)));
    $("preparation-counts").textContent = development >= 3 && final >= 3
      ? `${train} training and ${validation} validation experiments. ${final - 2} separate experiments are reserved for final testing.`
      : "";
  }
  function read(storageChanged = false) {
    const draft = drafts.get(key);
    try {
      const stored = localStorage.getItem(key);
      // After a failed write or removal, the old stored value still belongs to this draft.
      const neverRead = draft?.pending && draft.stored === undefined;
      if (draft && (draft.stored === stored || (neverRead && !storageChanged))) return draft.plan;
      const savedPlan = stored === null ? null : JSON.parse(stored);
      drafts.set(key, {plan: savedPlan, stored});
      return savedPlan;
    } catch {
      return draft?.plan ?? null;
    }
  }
  function save() {
    try {
      const stored = plan ? JSON.stringify(plan) : null;
      if (plan) localStorage.setItem(key, stored);
      else localStorage.removeItem(key);
      drafts.set(key, {plan, stored});
    } catch {
      drafts.set(key, {plan, stored: drafts.get(key)?.stored, pending: true});
    }
  }
  function show(message = "", error = false) {
    const ready = Boolean(plan?.benchmark);
    const review = Boolean(inventory) && !ready;
    $("preparation-size").disabled = Boolean(plan) || busy;
    $("preparation-review").hidden = !review;
    $("preparation-inventory").textContent = review
      ? `The service-wide inventory has ${inventory.dataset_count} registered datasets. Confirm its full scope to register the development dataset and reserve the final test partition.` : "";
    $("preparation-source-count").textContent = review ? `Review ${inventory.source_namespaces.length} source namespaces` : "";
    $("preparation-datasets").textContent = review ? `Development: ${plan.outputs.development} · Final: ${plan.outputs.final}` : "";
    $("preparation-sources").textContent = review ? inventory.source_namespaces.join("\n") : "";
    check.disabled = busy;
    action.disabled = busy || (review && !check.checked);
    action.textContent = ready ? "Use prepared data" : review ? "Reserve benchmark" :
      busy ? "Preparing…" : plan ? "Continue preparation" : "Generate datasets";
    const forget = $("preparation-forget");
    forget.hidden = !plan || busy;
    const messages = [message];
    if (drafts.get(key)?.pending) messages.push(plan
      ? "Progress is available only while this page stays open."
      : "The saved setup could not be removed from this browser. It may return after a reload.");
    status.textContent = messages.filter(Boolean).join(" ");
    status.hidden = !status.textContent;
    status.classList.toggle("error", error);
  }

  function open() {
    close();
    key = "regimex.training-preparation.v1." + session.value.principal.owner_id;
    plan = read();
    form.reset();
    if (plan) for (const id of ["development", "final", "steps"]) $("preparation-" + id).value = plan[id];
    panel.open = Boolean(plan);
    counts();
    show(plan?.benchmark ? "Datasets and benchmark ready." : plan ? "You have a saved preparation in this browser." : "");
  }
  function close() {
    ++revision;
    clearTimeout(timer);
    plan = inventory = null;
    key = "";
    busy = false;
    retryDelay = 1500;
    check.checked = false;
    panel.open = false;
    form.reset();
    show();
  }
  function active(stamp) { return revision === stamp && key && session.value?.principal.role === "admin"; }
  async function advance(stamp) {
    let reading = false;
    try {
      while (active(stamp)) {
        const index = stages.findIndex(([id]) => !plan.outputs[id]);
        if (index === -1) {
          reading = true;
          const query = new URLSearchParams({dataset_id: plan.outputs.development, final_dataset_id: plan.outputs.final});
          const reviewed = await session.json(api + "training-options/inventory?" + query);
          if (!active(stamp)) return;
          inventory = reviewed;
          busy = false;
          check.checked = false;
          show("Automatic labels are ready. Review the inventory before reserving the benchmark.");
          return;
        }
        const [id, label, output] = stages[index];
        show(`Step ${index + 1} of ${stages.length}: ${label}.`);
        let job;
        reading = Boolean(plan.job);
        if (plan.job) job = await session.json(api + "admin-runs/" + encodeURIComponent(plan.job));
        else {
          const dataset = id === "development" || id === "final";
          const body = dataset ? generationBody(plan, id === "final") :
            {mode: "automatic", dataset_id: plan.outputs[id === "test" ? "final" : "development"], partition: id};
          job = await session.json(api + (dataset ? "datasets" : "reference-sets"), {
            method: "POST", body, headers: {"Idempotency-Key": `${plan.id}:${id}:${plan.attempt}`},
          });
        }
        if (!active(stamp)) return;
        reading = false;
        retryDelay = 1500;
        if (job.job.state === "failed" || job.job.state === "cancelled") {
          delete plan.job;
          plan.attempt++;
          save();
          throw new Error(`${label}: ${job.job.failure?.message || "The job was cancelled."}`);
        }
        if (job.job.state === "succeeded") {
          if (!job.output?.[output]) throw new Error("This job completed without a resource ID.");
          plan.outputs[id] = job.output[output];
          delete plan.job;
          plan.attempt = 0;
          save();
        } else {
          plan.job = job.run_id;
          save();
          timer = setTimeout(() => void advance(stamp), 1500);
          return;
        }
      }
    } catch (error) {
      if (!active(stamp)) return;
      if (reading && isTransientRequestError(error)) {
        show("Could not update the preparation. Retrying automatically…");
        timer = setTimeout(() => void advance(stamp), retryDelay);
        retryDelay = Math.min(30000, retryDelay * 2);
      } else { busy = false; show(error.message, true); }
    }
  }
  async function submit(event) {
    event.preventDefault();
    if (busy || !key) return;
    const stamp = ++revision;
    busy = true;
    try {
      if (!plan) {
        const create = () => {
          if (!active(stamp)) return;
          if (read()) { open(); return; }
          const [devSeed, other] = crypto.getRandomValues(new Uint32Array(2));
          plan = {...fields(), devSeed, finalSeed: other === devSeed ? (other + 1) >>> 0 : other,
            id: crypto.randomUUID(), outputs: {}, attempt: 0};
          save();
        };
        // Two tabs starting together must agree on the idempotency keys before either submits work.
        if (navigator.locks) await navigator.locks.request(key, create);
        else create();
        if (!active(stamp)) return;
      }
      if (plan.benchmark) {
        show("Selecting the prepared dataset…");
        const used = await onUse(plan.outputs, plan.benchmark);
        if (!active(stamp)) return;
        if (!used) {
          busy = false;
          show("The form changed. Choose Use prepared data to select the dataset.");
          return;
        }
        plan = inventory = null;
        save();
        open();
        $("training-dataset").focus();
      } else if (inventory && check.checked) {
        show("Reserving the benchmark…");
        const result = await session.json(api + "training-options/selection", {
          method: "POST", body: {dataset_id: plan.outputs.development, final_dataset_id: plan.outputs.final,
            final_reference_set_id: plan.outputs.test, expected_registry_revision: inventory.revision,
            inventory_statement: inventoryStatement},
        });
        if (!active(stamp)) return;
        plan.benchmark = result.benchmark_id;
        save();
        busy = false;
        show("Datasets and benchmark ready.");
      } else {
        await advance(stamp);
      }
    } catch (error) {
      if (!active(stamp)) return;
      busy = false;
      inventory = null;
      check.checked = false;
      show(error.message, true);
    }
  }
  form.addEventListener("submit", submit);
  window.addEventListener("storage", event => {
    if (key && (event.key === key || event.key === null) && read(true) !== plan) open();
  });
  $("preparation-size").addEventListener("input", counts);
  check.addEventListener("change", () => { action.disabled = !check.checked; });
  $("preparation-forget").addEventListener("click", () => {
    if (busy || !plan || !window.confirm("Dismiss this setup? Any jobs and datasets already created will remain on the service.")) return;
    plan = null;
    save();
    open();
    panel.open = true;
  });
  return {open, close};
}
