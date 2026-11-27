import {$, addFieldHelp, option} from "./ui.js";
import {parseTrainingJSON, readTrainingRequest, trainingFieldsBody, trainingRequestKey} from "./training-json.js";
import {createTrainingPolicy, policyDraft, validateTrainingLimits} from "./training-policy.js";
import {createSizePicker, validateModelFields} from "./training-size.js";
import {createDataPreparation} from "./training-preparation.js";
import {architectureName, availableArchitectures, canonicalArchitectureId} from "./training-architectures.js";

const selectionFields = {
  "training-dataset": "dataset_id", "training-architecture": "architecture_id",
  "training-labels": "labels_reference_set_id", "training-validation": "validation_reference_set_id",
  "training-benchmark": "benchmark_selection_id",
};
const selectors = Object.keys(selectionFields);
const runtimes = ["training-runtime", "training-inference"];
const api = "/api/v1/";

export function shortId(id) {
  return id.replace(/^(?:dataset|reference)-([0-9a-f]{8})[0-9a-f]{24}$/, "$1");
}

export function datasetName(id) {
  const key = shortId(id);
  return key === id ? id : "Dataset\u00a0" + key;
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

function readSettings(id) {
  const field = $(id), input = field.value.trim();
  if (!input) return undefined;
  try {
    const value = JSON.parse(input);
    if (value && typeof value === "object" && !Array.isArray(value)) return input;
  } catch {
    // Use the same field-level message for malformed JSON and the wrong JSON type.
  }
  $("training-advanced-settings").open = true;
  field.setAttribute("aria-invalid", "true");
  field.focus();
  throw new Error("Enter a JSON object for " + field.labels[0].firstChild.textContent.trim() + ".");
}

function formBody(name, selections, profile, {policy, parameters, training_runtime, inference_runtime}) {
  return trainingFieldsBody({display_name: name?.trim() ? name : undefined, ...selections,
    analysis_profile_id: profile}, {policy, parameters, training_runtime, inference_runtime});
}

export function createTrainingForm({session, eachPage}) {
  let generation = 0, busy = false, architectures = new Map(), draft = null;
  let currentArchitecture = null;
  let sourceConfiguration = null;
  let jsonBase = null, importedBody = null;
  let converting = false, loading = false, analysisProfile;
  const jsonField = $("training-json");
  const architectureDrafts = new Map();
  const policy = createTrainingPolicy($("training-policy"));
  const sizePicker = createSizePicker();
  const preparation = createDataPreparation({session, onUse: usePreparedData});
  const hint = $("training-prerequisite");
  function architectureFor(id) {
    return architectures.get(canonicalArchitectureId(id));
  }
  function resolveArchitectureId(id) {
    return architectureFor(id)?.architecture_id ?? id;
  }
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
    const locked = busy || converting;
    $("training-edit-mode").disabled = locked || loading;
    $("training-discard-json").disabled = locked;
    $("training-json").readOnly = locked;
    $("training-submit").disabled = locked || (!jsonBase && selectors.some(id => $(id).disabled || !$(id).value));
  }
  function saveArchitecture() {
    if (currentArchitecture) architectureDrafts.set(currentArchitecture, {
      policy: policy.snapshot(), parameters: $("training-parameters").value,
    });
  }
  function selectArchitecture() {
    saveArchitecture();
    const architecture = architectureFor($("training-architecture").value);
    currentArchitecture = architecture?.architecture_id;
    const saved = architectureDrafts.get(currentArchitecture);
    policy.select(architecture, saved?.policy);
    $("training-parameters").value = saved?.parameters ?? "{}";
    sizePicker.selectArchitecture(architecture);
    $("training-parameters").removeAttribute("aria-invalid");
  }
  const dependent = ["training-labels", "training-validation", "training-benchmark"];
  function restoreChoice(id, saved) {
    if (saved != null) $(id).value = saved;
  }
  async function datasetOptions(id, valid) {
    const query = "?dataset_id=" + encodeURIComponent(id);
    const [references, benchmarks] = await Promise.all([
      eachPage(api + "reference-sets" + query, valid), session.json(api + "training-options/benchmarks" + query),
    ]);
    const referencesFor = partition => references
      .filter(row => row.partition === partition && row.current_training_profile)
      .map(row => [row.reference_set_id, [
        row.kind === "automatic" ? "Automatic" : "Human", row.reviewer, shortId(row.reference_set_id),
      ].filter(Boolean).join(" · ")]);
    return {
      profile: benchmarks.analysis_profile_id,
      "training-labels": referencesFor("train"), "training-validation": referencesFor("validation"),
      "training-benchmark": benchmarks.benchmark_ids.map(id => [id, id]),
    };
  }
  function setDatasetOptions(options) {
    for (const id of dependent)
      choices(id, options[id], id === "training-benchmark" ? "No eligible benchmarks" : "No matching references");
  }
  async function loadDataset() {
    const stamp = ++generation, valid = () => generation === stamp;
    const id = $("training-dataset").value;
    loading = Boolean(id);
    feedback();
    for (const key of dependent) choices(key, [], id ? "Loading…" : "Choose a dataset first");
    updateSubmit();
    if (!id) {
      feedback("Choose Prepare training data to generate your first development dataset.");
      return;
    }
    try {
      const options = await datasetOptions(id, valid);
      if (!valid()) return;
      setDatasetOptions(options);
      if (draft?.["training-dataset"] === id)
        for (const field of dependent) restoreChoice(field, draft[field]);
      if (!$("training-labels").value || !$("training-validation").value)
        feedback("This dataset needs compatible references for both training and validation.",
          ["Create references", "create_references_api_v1_reference_sets_post"]);
      else if (!options["training-benchmark"].length)
        feedback("This dataset has no benchmark open for model selection.",
          ["Reserve a benchmark", "reserve_api_v1_benchmarks_post"],
          ["Start benchmark selection", "selection_start_api_v1_benchmarks_selection_start_post"]);
      else if (!architectures.size) feedback("Install a training backend to start a run.");
    } catch (error) {
      if (valid()) feedback(error.message, ["Retry", loadDataset]);
    }
    if (valid()) { loading = false; updateSubmit(); }
  }
  async function usePreparedData(outputs, benchmark) {
    if (jsonBase) throw new Error("Switch to the form to select the prepared data.");
    const stamp = ++generation, valid = () => generation === stamp;
    const [dataset, options] = await Promise.all([
      session.json(api + "datasets/" + encodeURIComponent(outputs.development)),
      datasetOptions(outputs.development, valid),
    ]);
    if (!valid() || jsonBase) return false;
    const selected = {"training-labels": outputs.train, "training-validation": outputs.validation,
      "training-benchmark": benchmark};
    if (dependent.some(id => !options[id].some(([value]) => value === selected[id])))
      throw new Error("The prepared references or benchmark are no longer available for training.");
    const picker = $("training-dataset");
    if (!Array.from(picker.options).some(row => row.value === dataset.dataset_id)) {
      if (picker.disabled) picker.replaceChildren();
      picker.add(option(dataset.dataset_id, datasetName(dataset.dataset_id) + " · " +
        dataset.partitions.train + " training · " + dataset.partitions.validation + " validation"));
    }
    picker.disabled = false;
    picker.value = dataset.dataset_id;
    setDatasetOptions(options);
    for (const id of dependent) $(id).value = selected[id];
    loading = false;
    feedback();
    updateSubmit();
    picker.dispatchEvent(new Event("input", {bubbles: true}));
    return true;
  }
  async function open() {
    for (const label of $("training-fields").querySelectorAll("label[data-help]")) {
      addFieldHelp(label, label.control, label.dataset.help);
      delete label.dataset.help;
    }
    const stamp = ++generation, valid = () => generation === stamp;
    if ($("training-config").hidden) {
      $("training-name").value = draft?.["training-name"] || "";
      analysisProfile = draft?.analysisProfile;
      importedBody = draft?.importedBody;
      if (draft?.jsonBase) showEditor(draft.jsonBase, draft.jsonText);
    }
    $("training-config").hidden = false;
    preparation.open();
    loading = true;
    feedback();
    for (const id of selectors) choices(id, [], "Loading…");
    updateSubmit();
    try {
      const [datasets, backends] = await Promise.all([
        eachPage(api + "datasets?development_only=true", valid,
          page => preparation.setExperimentLimit(page.max_generation_experiments)),
        eachPage(api + "architectures", valid),
      ]);
      if (!valid()) return;
      choices("training-dataset", datasets.filter(row => row.partitions.train && row.partitions.validation)
        .map(row => [row.dataset_id, datasetName(row.dataset_id) + " · " + row.partitions.train +
          " training · " + row.partitions.validation + " validation"]), "No development datasets");
      architectures = availableArchitectures(backends);
      choices("training-architecture", Array.from(architectures.values(),
        row => [row.architecture_id, architectureName(row.architecture_id)]), "No training backends");
      restoreChoice("training-dataset", draft?.["training-dataset"]);
      restoreChoice("training-architecture", resolveArchitectureId(draft?.["training-architecture"]));
      selectArchitecture();
      if (sourceConfiguration) {
        const source = sourceConfiguration;
        try {
          await applyRequest(source, source, true);
        } catch (error) {
          if (!valid()) return;
          // Keep the saved settings editable when the current form cannot represent them.
          const fallback = loadDataset(), fallbackStamp = generation;
          await fallback;
          if (generation !== fallbackStamp) return;
          sourceConfiguration = null;
          showEditor(fieldsBody(), source);
          feedback(error.message);
          markJsonError("training-prerequisite");
          return;
        }
        if (valid()) {
          sourceConfiguration = null;
          loading = false;
          updateSubmit();
        }
        return;
      }
      if (draft) {
        for (const id of runtimes) $(id).value = draft[id] ?? "";
        $("training-form-settings").open = draft.policyExpanded;
        $("training-advanced-settings").open = draft.advancedExpanded;
      }
      await loadDataset();
    } catch (error) {
      if (valid()) { loading = false; updateSubmit(); feedback(error.message, ["Retry", open]); }
    }
  }
  function close(remember = true) {
    preparation.close();
    if (!remember) {
      draft = null;
      sourceConfiguration = null;
      architectureDrafts.clear();
    } else if (!$("training-config").hidden) {
      if (currentArchitecture) {
        saveArchitecture();
        draft = Object.fromEntries([...selectors, ...runtimes].map(id => [id, $(id).value]));
        draft.policyExpanded = $("training-form-settings").open;
        draft.advancedExpanded = $("training-advanced-settings").open;
      }
      draft = {...draft, "training-name": $("training-name").value, analysisProfile,
        importedBody, jsonBase, jsonText: jsonField.value};
    }
    generation++;
    architectures.clear();
    loading = false;
    converting = false;
    analysisProfile = undefined;
    importedBody = null;
    showEditor(null);
    currentArchitecture = null;
    policy.select(null);
    sizePicker.selectArchitecture(null);
    $("training-name").value = "";
    $("training-config").hidden = true;
    $("training-form-settings").open = false;
    $("training-advanced-settings").open = false;
    for (const id of ["training-parameters", ...runtimes]) {
      $(id).value = "";
      $(id).removeAttribute("aria-invalid");
    }
    for (const id of selectors) $(id).replaceChildren();
    feedback();
  }
  $("training-dataset").addEventListener("change", loadDataset);
  $("training-architecture").addEventListener("change", selectArchitecture);
  for (const id of dependent) $(id).addEventListener("change", updateSubmit);
  $("training-policy").addEventListener("invalid", () => { $("training-form-settings").open = true; }, true);
  for (const id of ["training-parameters", ...runtimes])
    $(id).addEventListener("input", () => $(id).removeAttribute("aria-invalid"));
  function formJSON() {
    sizePicker.validate();
    const parameters = readSettings("training-parameters") ?? "{}";
    try {
      validateModelFields(architectureFor(currentArchitecture), parameters);
    } catch (error) {
      $("training-advanced-settings").open = true;
      $("training-parameters").setAttribute("aria-invalid", "true");
      $("training-parameters").focus();
      throw error;
    }
    const policyFields = currentArchitecture ? policy.payload() : undefined;
    const originalPolicy = policyFields && importedBody?.architecture === currentArchitecture &&
      importedBody.policyKey === trainingRequestKey(policyFields) ? importedBody.policyText : undefined;
    const selections = Object.fromEntries(selectors.map(id => [selectionFields[id], $(id).value]));
    return formBody($("training-name").value, selections, analysisProfile, {
      policy: originalPolicy ?? JSON.stringify(policyFields),
      parameters,
      training_runtime: readSettings("training-runtime"),
      inference_runtime: readSettings("training-inference"),
    });
  }
  function showEditor(base, text = base ?? "") {
    jsonBase = base;
    $("training-fields").hidden = Boolean(base);
    $("training-fields").disabled = Boolean(base);
    $("training-json-panel").hidden = !base;
    $("training-edit-mode").textContent = base ? "Use form" : "Edit JSON";
    jsonField.value = text;
    jsonField.removeAttribute("aria-invalid");
    jsonField.removeAttribute("aria-errormessage");
    $("training-discard-json").hidden = !base || text === base;
    updateSubmit();
  }
  function markJsonError(messageId) {
    jsonField.setAttribute("aria-invalid", "true");
    jsonField.setAttribute("aria-errormessage", messageId);
    jsonField.focus();
  }
  async function useForm() {
    if (jsonField.value === jsonBase) {
      showEditor(null);
      $("training-name").focus();
      return;
    }
    await applyRequest(jsonField.value, jsonBase);
  }
  async function applyRequest(text, base, fromRun = false) {
    const {request, settings, body} = parseTrainingJSON(text, base, {fromRun, resolveArchitectureId});
    if (/[\r\n]/.test(request.display_name ?? "")) throw new Error("Run name must be on one line.");
    const architecture = architectureFor(request.architecture_id);
    if (!architecture) throw new Error("The selected architecture is unavailable in this workspace.");
    const datasetReady = Array.from($("training-dataset").options).some(item => item.value === request.dataset_id);
    if (!datasetReady && !fromRun)
      throw new Error("Choose a development dataset available in this workspace.");
    const saved = currentArchitecture === request.architecture_id
      ? policy.snapshot() : architectureDrafts.get(request.architecture_id)?.policy;
    const imported = policyDraft(architecture, request.policy, saved);
    validateModelFields(architecture, settings.parameters);
    const stamp = generation, valid = () => generation === stamp;
    converting = true;
    updateSubmit();
    try {
      const options = datasetReady ? await datasetOptions(request.dataset_id, valid)
        : Object.fromEntries(dependent.map(id => [id, []]));
      if (!valid()) return;
      const selections = {dataset_id: datasetReady ? request.dataset_id : "", architecture_id: request.architecture_id};
      for (const id of dependent) {
        const key = selectionFields[id];
        selections[key] = options[id].some(([value]) => value === request[key]) ? request[key] : "";
        if (!fromRun && !selections[key]) throw new Error(key + " is unavailable for this dataset.");
      }
      const changedProfile = fromRun && options.profile && request.analysis_profile_id &&
        request.analysis_profile_id !== options.profile;
      const profile = fromRun && !datasetReady ? undefined :
        changedProfile ? options.profile : request.analysis_profile_id;
      const missing = Object.values(selections).some(value => !value);
      // Preserve sparse policies and exact number literals until a field is edited.
      const fields = formBody(request.display_name, selections, profile, settings);
      const nextBody = {architecture: request.architecture_id, policyKey: trainingRequestKey(imported.policy),
        policyText: settings.policy, fields, text: missing || changedProfile ? fields : body};
      saveArchitecture();
      setDatasetOptions(options);
      for (const id of selectors) $(id).value = selections[selectionFields[id]];
      currentArchitecture = request.architecture_id;
      policy.select(architecture, imported.draft);
      $("training-name").value = request.display_name ?? "";
      $("training-parameters").value = settings.parameters;
      sizePicker.selectArchitecture(architecture);
      $("training-runtime").value = settings.training_runtime ?? "";
      $("training-inference").value = settings.inference_runtime ?? "";
      analysisProfile = profile;
      importedBody = nextBody;
      // An unavailable reference or closed reservation must be chosen again before submission.
      if (missing || changedProfile) {
        feedback([changedProfile && `New runs use ${options.profile} instead of ${request.analysis_profile_id}.`,
          missing && "Choose the missing dataset, labels or benchmark to continue."].filter(Boolean).join(" "));
      }
      showEditor(null);
      $("training-name").focus();
    } catch (error) {
      if (valid()) throw error;
    } finally {
      if (valid()) { converting = false; updateSubmit(); }
    }
  }
  $("training-edit-mode").addEventListener("click", async () => {
    feedback();
    try {
      if (jsonBase) await useForm();
      else {
        const body = fieldsBody();
        showEditor(body);
        jsonField.focus();
      }
    } catch (error) {
      feedback(error.message);
      if (jsonBase) markJsonError("training-prerequisite");
    }
  });
  jsonField.addEventListener("input", () => {
    jsonField.removeAttribute("aria-invalid");
    jsonField.removeAttribute("aria-errormessage");
    $("training-discard-json").hidden = jsonField.value === jsonBase;
    feedback();
  });
  $("training-discard-json").addEventListener("click", () => {
    showEditor(null);
    feedback();
    $("training-name").focus();
  });
  function fieldsBody() {
    const fields = formJSON();
    return importedBody?.fields === fields ? importedBody.text : fields;
  }
  function payload() {
    if (!jsonBase) return fieldsBody();
    try {
      const {request, body} = readTrainingRequest(jsonField.value, jsonBase, {resolveArchitectureId});
      const architecture = architectureFor(request.architecture_id);
      if (architecture) validateTrainingLimits(architecture, request.policy);
      return body;
    } catch (error) {
      markJsonError("training-notice");
      throw error;
    }
  }
  return {
    open, close, payload,
    useSavedRun(configuration) { sourceConfiguration = configuration; },
    discard() { close(false); },
    setBusy(value) { busy = value; updateSubmit(); },
  };
}
