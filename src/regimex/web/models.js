import {$, option} from "./ui.js";

const MODEL_PAGE_SIZE = 100;

export function createModels({session, getSelection, getBusy, getExport}) {
  let models = [];
  let activeModelId = null;
  let modelOffset = 0;
  let modelHasMore = false;
  let modelGeneration = 0;
  let modelsReady = false;
  let modelsLoading = false;
  function reset() {
    modelGeneration++;
    models = [];
    activeModelId = null;
    modelOffset = 0;
    modelHasMore = modelsReady = modelsLoading = false;
    $("upload-model").replaceChildren();
    $("rerun-model").replaceChildren();
    $("upload-model").value = $("rerun-model").value = "";
    $("rerun-controls").hidden = true;
  }

  function fillModelSelect(name, preferred = null) {
    const select = $(name);
    const selected = preferred ?? select.value;
    const choices = models.map((model) => [model.model_version_id, model.model_version_id]);
    // A previously selected published version remains selectable while browsing another page.
    if (selected && selected !== "active" && !choices.some(([id]) => id === selected))
      choices.unshift([selected, selected]);
    if (activeModelId) choices.unshift(["active", `Active model · ${activeModelId}`]);
    select.replaceChildren(option("", "Choose a published model"));
    choices.forEach(([id, label]) => select.append(option(id, label)));
    select.value = choices.some(([id]) => id === selected)
      ? selected : activeModelId ? "active" : "";
  }

  function updateActions() {
    $("classify").disabled = getBusy() || !session.value || !getExport() || !modelsReady || !$("upload-model").value;
    $("export-file").disabled = getBusy();
    $("upload-model").disabled = getBusy() || modelsLoading || !modelsReady;
    $("rerun-model").disabled = getBusy() || modelsLoading || !modelsReady;
    const saved = getSelection()?.metadata;
    const terminal = saved && ["succeeded", "failed", "cancelled"].includes(saved.job.state);
    $("rerun-controls").hidden = !terminal || !saved.snapshot_id;
    $("rerun").disabled = getBusy() || !terminal || !modelsReady || !$("rerun-model").value;
    ["models-previous", "rerun-models-previous"].forEach((id) => {
      $(id).hidden = modelOffset === 0;
      $(id).disabled = getBusy() || modelsLoading;
    });
    ["models-next", "rerun-models-next"].forEach((id) => {
      $(id).hidden = !modelHasMore;
      $(id).disabled = getBusy() || modelsLoading;
    });
    $("models-refresh").disabled = getBusy() || modelsLoading;
  }

  function modelStatus(message) {
    $("model-status").textContent = $("rerun-model-status").textContent = message;
  }

  async function loadModels(offset = 0) {
    const workspace = session.epoch;
    const generation = ++modelGeneration;
    modelsLoading = true;
    modelStatus("Loading models…");
    updateActions();
    try {
      const data = await session.json(`/api/v1/models?published_only=true&limit=${MODEL_PAGE_SIZE}&offset=${offset}`);
      if (workspace !== session.epoch || generation !== modelGeneration || !session.value) return;
      models = data.items;
      activeModelId = data.active_model_version_id;
      modelOffset = offset;
      modelHasMore = models.length === MODEL_PAGE_SIZE;
      modelsReady = true;
      fillModelSelect("upload-model");
      fillModelSelect("rerun-model", $("rerun-model").value || getSelection()?.metadata?.job.model?.model_version_id);
      modelStatus(models.length || activeModelId ? "" : offset
        ? "No more models. Return to the previous page."
        : "No published models are available. Ask your administrator to publish a model.");
    } catch (error) {
      if (workspace === session.epoch && generation === modelGeneration) modelStatus(error.message);
    } finally {
      if (workspace === session.epoch && generation === modelGeneration) {
        modelsLoading = false;
        updateActions();
      }
    }
  }

  function selectedModel(name) {
    return $(name).value === "active" ? activeModelId : $(name).value;
  }
  $("upload-model").addEventListener("change", updateActions);
  $("rerun-model").addEventListener("change", updateActions);
  ["models-previous", "rerun-models-previous"].forEach((id) =>
    $(id).addEventListener("click", () => loadModels(Math.max(0, modelOffset - MODEL_PAGE_SIZE))));
  ["models-next", "rerun-models-next"].forEach((id) =>
    $(id).addEventListener("click", () => loadModels(modelOffset + MODEL_PAGE_SIZE)));
  $("models-refresh").addEventListener("click", () => loadModels());
  return {reset, load: loadModels, fillSelect: fillModelSelect, updateActions,
    selected: selectedModel, get ready() { return modelsReady; }};
}
