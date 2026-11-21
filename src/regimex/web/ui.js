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

// A navigation inside a mounted iframe would create a browser history entry of its own.
export function setPreview(id, path = null) {
  const frame = $(id);
  frame.hidden = !path;
  if (frame.getAttribute("src") === path) return;
  const parent = frame.parentNode, next = frame.nextSibling;
  if (parent) frame.remove();
  if (path) frame.setAttribute("src", path);
  else frame.removeAttribute("src");
  if (parent) parent.insertBefore(frame, next);
}
