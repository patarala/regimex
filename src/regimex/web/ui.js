export const $ = (id) => document.getElementById(id);

const analysisStates = {
  queued: "Queued", running: "Running", cancel_requested: "Cancelling",
  cancelled: "Cancelled", succeeded: "Complete", failed: "Failed",
};

export function analysisStateLabel(state) {
  return analysisStates[state] || state;
}

export function shortDate(value) {
  return new Date(value).toLocaleString(undefined, {dateStyle: "medium", timeStyle: "short"});
}

export function notice(message = "", error = false) {
  $("notice").textContent = message;
  $("notice").classList.toggle("error", error);
  $("notice").hidden = !message;
}

export function element(tag, text = "", className = "") {
  const el = document.createElement(tag);
  el.textContent = text;
  if (className) el.className = className;
  return el;
}

export function option(value, label) {
  const item = element("option", label);
  item.value = value;
  return item;
}

export function addFieldHelp(label, field, text) {
  const heading = element("div", "", "field-heading");
  const note = element("p", text, "field-help");
  note.id = field.id + "-help";
  note.hidden = true;
  const button = element("button", "", "field-help-toggle");
  button.type = "button";
  button.setAttribute("aria-controls", note.id);
  button.setAttribute("aria-expanded", "false");
  const icon = element("span", "?");
  icon.setAttribute("aria-hidden", "true");
  button.append(icon, element("span", "Help for " + label.textContent, "visually-hidden"));
  label.replaceWith(heading);
  heading.append(label, button);
  field.after(note);
  field.setAttribute("aria-describedby", note.id);

  function setOpen(open) {
    note.hidden = !open;
    button.setAttribute("aria-expanded", String(open));
  }
  button.addEventListener("click", () => setOpen(note.hidden));
  heading.parentElement.addEventListener("keydown", event => {
    if (event.key === "Escape" && !note.hidden) {
      setOpen(false);
      button.focus();
      event.stopPropagation();
    }
  });
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
