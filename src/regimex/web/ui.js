export const $ = (id) => document.getElementById(id);

const analysisStates = {
  queued: "Queued", running: "Running", cancel_requested: "Cancelling",
  cancelled: "Cancelled", succeeded: "Complete", failed: "Failed",
};

export function analysisStateLabel(state) {
  return analysisStates[state] || state;
}

export function notice(message = "", error = false) {
  $("notice").textContent = message;
  $("notice").classList.toggle("error", error);
  $("notice").hidden = !message;
}

export function option(value, label) {
  const item = document.createElement("option");
  item.value = value;
  item.textContent = label;
  return item;
}
