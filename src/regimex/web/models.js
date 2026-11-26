import {$, option} from "./ui.js";

const MODEL_PAGE_SIZE = 50;

export function createModels({session, getSelection, getBusy, getUploadReady, getBatchLocked}) {
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
    const placeholder = choices.length ? "Choose a published model"
      : modelOffset ? "No models on this page" : "No published models";
    select.replaceChildren(option("", placeholder));
    choices.forEach(([id, label]) => select.append(option(id, label)));
    select.value = choices.some(([id]) => id === selected)
      ? selected : activeModelId ? "active" : "";
  }

  function updateActions() {
    $("classify").disabled = getBusy() || !session.value || !getUploadReady() || !modelsReady || !$("upload-model").value;
    $("export-file").disabled = getBusy() || getBatchLocked();
    const unavailable = getBusy() || modelsLoading || !modelsReady;
    $("upload-model").disabled = unavailable || getBatchLocked() || $("upload-model").children.length < 2;
    $("rerun-model").disabled = unavailable || $("rerun-model").children.length < 2;
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

  function modelStatus(message, linkToTraining = false) {
    for (const id of ["model-status", "rerun-model-status"]) {
      $(id).textContent = message;
      if (linkToTraining) {
        const link = document.createElement("a");
        link.href = "/training";
        link.dataset.workspaceLink = "";
        link.textContent = "Open Training";
        $(id).append(link);
      }
    }
  }

  async function loadModels(offset = 0) {
    const workspace = session.epoch;
    const generation = ++modelGeneration;
    modelsLoading = true;
    modelStatus("Loading models…");
    updateActions();
    try {
      const data = await session.json(`/api/v1/models?published_only=true&limit=${MODEL_PAGE_SIZE + 1}&offset=${offset}`);
      if (workspace !== session.epoch || generation !== modelGeneration || !session.value) return;
      models = data.items.slice(0, MODEL_PAGE_SIZE);
      activeModelId = data.active_model_version_id;
      modelOffset = offset;
      modelHasMore = data.items.length > MODEL_PAGE_SIZE;
      modelsReady = true;
      fillModelSelect("upload-model");
      fillModelSelect("rerun-model", $("rerun-model").value || getSelection()?.metadata?.job.model?.model_version_id);
      const empty = !models.length && !activeModelId;
      const admin = session.value.principal.role === "admin";
      modelStatus(!empty ? "" : offset ? "Return to the previous page for more models."
        : admin ? "Publish a model to start an analysis. " : "Ask your administrator to publish a model.",
        empty && !offset && admin);
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
