import {$, option} from "./ui.js";

const selectors = ["training-dataset", "training-architecture", "training-labels", "training-validation", "training-benchmark"];
const settings = ["training-parameters", "training-policy", "training-runtime", "training-inference"];
const api = "/api/v1/";

export function architectureName(id, compact = false) {
  if (id === "tcn") return compact ? "TCN" : "Temporal convolutional network (TCN)";
  if (id === "baseline") return compact ? "Baseline" : "Logistic baseline";
  return id;
}

function shortId(id) {
  return id.replace(/^(?:dataset|reference)-([0-9a-f]{8})[0-9a-f]{24}$/, "$1");
}

function choices(id, items, empty) {
  const select = $(id);
  select.replaceChildren(...(items.length ? items.map(([value, label]) => {
    const item = option(value, label);
    item.title = value;
    return item;
  }) : [option("", empty)]));
  select.disabled = !items.length;
}

function readSettings(id, optional = false) {
  const field = $(id), input = field.value.trim();
  if (!input && optional) return undefined;
  try {
    const value = JSON.parse(input);
    if (value && typeof value === "object" && !Array.isArray(value)) return value;
  } catch {
    // Use the same field-level message for malformed JSON and the wrong JSON type.
  }
  $("training-form-settings").open = true;
  field.setAttribute("aria-invalid", "true");
  field.focus();
  throw new Error("Enter a JSON object for " + field.labels[0].firstChild.textContent.trim() + ".");
}

export function createTrainingForm({session, eachPage}) {
  let generation = 0, busy = false, architectures = [], draft = null;
  const hint = $("training-prerequisite");
  function feedback(message = "", ...actions) {
    hint.replaceChildren();
    hint.hidden = !message;
    if (!message) return;
    hint.append(message);
    for (const [action, handler] of actions) {
      const retry = typeof handler === "function";
      const el = document.createElement(retry ? "button" : "a");
      el.textContent = action;
      if (retry) {
        el.type = "button";
        el.className = "quiet";
        el.addEventListener("click", handler);
      } else {
        el.href = "/docs#/default/" + handler;
      }
      hint.append(" ", el);
    }
  }
  function updateSubmit() {
    $("training-submit").disabled = busy || selectors.some(id => $(id).disabled || !$(id).value);
  }
  function policy() {
    const selected = architectures.find(row => row.architecture_id === $("training-architecture").value);
    $("training-policy").value = selected ? JSON.stringify(selected.default_training_policy, null, 2) : "";
    $("training-parameters").value = "{}";
    for (const id of settings.slice(0, 2)) $(id).removeAttribute("aria-invalid");
  }
  const dependent = ["training-labels", "training-validation", "training-benchmark"];
  function restoreChoice(id, saved) {
    if (saved && Array.from($(id).options).some(item => item.value === saved)) $(id).value = saved;
  }
  async function loadDataset() {
    const stamp = ++generation, valid = () => generation === stamp;
    const id = $("training-dataset").value;
    feedback();
    for (const key of dependent) choices(key, [], id ? "Loading…" : "Choose a dataset first");
    updateSubmit();
    if (!id) {
      feedback("Register a development dataset to start training.",
        ["Create dataset", "create_dataset_api_v1_datasets_post"],
        ["Register dataset", "register_dataset_api_v1_benchmarks_history_datasets_post"]);
      return;
    }
    try {
      const query = "?dataset_id=" + encodeURIComponent(id);
      const [references, benchmarks] = await Promise.all([
        eachPage(api + "reference-sets" + query, valid), session.json(api + "training-options/benchmarks" + query),
      ]);
      if (!valid()) return;
      for (const [field, partition] of [["training-labels", "train"], ["training-validation", "validation"]]) {
        const rows = references.filter(row => row.partition === partition && row.current_training_profile);
        choices(field, rows.map(row => [row.reference_set_id, [
          row.kind === "automatic" ? "Automatic" : "Human", row.reviewer, shortId(row.reference_set_id),
        ].filter(Boolean).join(" · ")]), "No matching references");
      }
      choices("training-benchmark", benchmarks.benchmark_ids.map(value => [value, value]), "No eligible benchmarks");
      if (draft?.["training-dataset"] === id)
        for (const field of dependent) restoreChoice(field, draft[field]);
      if (!$("training-labels").value || !$("training-validation").value)
        feedback("This dataset needs compatible references for both training and validation.",
          ["Create references", "create_references_api_v1_reference_sets_post"]);
      else if (!benchmarks.benchmark_ids.length)
        feedback("This dataset has no benchmark open for model selection.",
          ["Reserve a benchmark", "reserve_api_v1_benchmarks_post"],
          ["Start benchmark selection", "selection_start_api_v1_benchmarks_selection_start_post"]);
      else if (!architectures.length) feedback("Install a training backend to start a run.");
    } catch (error) {
      if (valid()) feedback(error.message, ["Retry", loadDataset]);
    }
    if (valid()) updateSubmit();
  }
  async function open() {
    const stamp = ++generation, valid = () => generation === stamp;
    $("training-config").hidden = false;
    feedback();
    for (const id of selectors) choices(id, [], "Loading…");
    updateSubmit();
    try {
      const [datasets, backends] = await Promise.all([
        eachPage(api + "datasets?development_only=true", valid), eachPage(api + "architectures", valid),
      ]);
      if (!valid()) return;
      choices("training-dataset", datasets.filter(row => row.partitions.train && row.partitions.validation)
        .map(row => [row.dataset_id, shortId(row.dataset_id) + " · " + row.partitions.train +
          " training · " + row.partitions.validation + " validation"]), "No development datasets");
      architectures = backends.filter(row => row.executable !== false);
      choices("training-architecture", architectures.map(row => [row.architecture_id, architectureName(row.architecture_id)]),
        "No training backends");
      restoreChoice("training-dataset", draft?.["training-dataset"]);
      restoreChoice("training-architecture", draft?.["training-architecture"]);
      policy();
      if (draft && draft["training-architecture"] === $("training-architecture").value) {
        for (const id of settings) $(id).value = draft[id];
        $("training-form-settings").open = draft.expanded;
      }
      await loadDataset();
    } catch (error) { if (valid()) feedback(error.message, ["Retry", open]); }
  }
  function close(remember = true) {
    if (!remember) draft = null;
    else if (!$("training-config").hidden && $("training-architecture").value) {
      draft = Object.fromEntries([...selectors, ...settings].map(id => [id, $(id).value]));
      draft.expanded = $("training-form-settings").open;
    }
    generation++;
    architectures = [];
    $("training-config").hidden = true;
    $("training-form-settings").open = false;
    for (const id of settings) {
      $(id).value = "";
      $(id).removeAttribute("aria-invalid");
    }
    for (const id of selectors) $(id).replaceChildren();
    feedback();
  }
  $("training-dataset").addEventListener("change", loadDataset);
  $("training-architecture").addEventListener("change", policy);
  for (const id of settings)
    $(id).addEventListener("input", () => $(id).removeAttribute("aria-invalid"));
  function payload() {
    return {
      dataset_id: $("training-dataset").value, architecture_id: $("training-architecture").value,
      labels_reference_set_id: $("training-labels").value, validation_reference_set_id: $("training-validation").value,
      benchmark_selection_id: $("training-benchmark").value, policy: readSettings("training-policy"),
      parameters: readSettings("training-parameters"), training_runtime: readSettings("training-runtime", true),
      inference_runtime: readSettings("training-inference", true),
    };
  }
  return {
    open, close, payload,
    discard() { close(false); },
    setBusy(value) { busy = value; updateSubmit(); },
  };
}
