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

let closeActiveHelp;

function positionHelp(button, note) {
  const anchor = button.getBoundingClientRect();
  const viewport = window.visualViewport;
  const left = viewport?.offsetLeft ?? 0, top = viewport?.offsetTop ?? 0;
  const width = viewport?.width ?? document.documentElement.clientWidth;
  const bottom = top + (viewport?.height ?? document.documentElement.clientHeight);
  if (!anchor.width || !anchor.height || anchor.bottom <= top || anchor.top >= bottom ||
      anchor.right <= left || anchor.left >= left + width) return false;

  const margin = 8;
  const below = bottom - anchor.bottom - margin, above = anchor.top - top - margin;
  note.style.maxWidth = `min(22rem, ${width - 2 * margin}px)`;
  note.style.maxHeight = Math.max(above, below, 0) + "px";
  const panel = note.getBoundingClientRect();
  const centered = anchor.left + (anchor.width - panel.width) / 2;
  note.style.left = Math.max(left + margin, Math.min(centered, left + width - margin - panel.width)) + "px";
  // Adjoining edges let the pointer move straight from the button onto the note.
  note.style.top = (below >= panel.height || below >= above ? anchor.bottom : anchor.top - panel.height) + "px";
  return true;
}

export function addFieldHelp(label, field, text) {
  const heading = element("div", "", "field-heading");
  const note = element("p", text, "field-help");
  note.id = field.id + "-help";
  note.setAttribute("role", "tooltip");
  note.setAttribute("popover", "manual");
  note.hidden = true;
  const button = element("button", "", "field-help-toggle");
  button.type = "button";
  button.setAttribute("aria-controls", note.id);
  button.setAttribute("aria-describedby", note.id);
  button.setAttribute("aria-expanded", "false");
  const icon = element("span", "?");
  icon.setAttribute("aria-hidden", "true");
  button.append(icon, element("span", "Help for " + label.textContent, "visually-hidden"));
  label.replaceWith(heading);
  heading.append(label, button);
  field.after(note);
  field.setAttribute("aria-describedby", note.id);

  let pinned = false, timer, observer, listeners;
  function close() {
    if (closeActiveHelp !== close) return;
    clearTimeout(timer);
    observer.disconnect();
    listeners.abort();
    if (note.hidePopover && note.matches(":popover-open")) note.hidePopover();
    note.hidden = true;
    pinned = false;
    button.setAttribute("aria-expanded", "false");
    closeActiveHelp = undefined;
  }
  function position() {
    if (!positionHelp(button, note)) close();
  }
  function show() {
    clearTimeout(timer);
    if (closeActiveHelp === close) return;
    closeActiveHelp?.();
    closeActiveHelp = close;
    listeners = new AbortController();
    const {signal} = listeners;
    note.hidden = false;
    note.showPopover?.();
    button.setAttribute("aria-expanded", "true");
    document.addEventListener("pointerdown", event => {
      if (!button.contains(event.target) && !note.contains(event.target)) close();
    }, {capture: true, signal});
    document.addEventListener("keydown", event => {
      if (event.key !== "Escape") return;
      if (document.activeElement === field) button.focus({preventScroll: true});
      close();
      event.stopPropagation();
    }, {capture: true, signal});
    document.addEventListener("focusin", event => {
      if (pinned && event.target !== field && event.target !== button && !note.contains(event.target)) close();
    }, {signal});
    document.addEventListener("scroll", position, {capture: true, passive: true, signal});
    window.addEventListener("resize", position, {signal});
    window.visualViewport?.addEventListener("resize", position, {signal});
    window.visualViewport?.addEventListener("scroll", position, {signal});
    window.addEventListener("hashchange", close, {signal});
    window.addEventListener("pagehide", close, {signal});
    observer = new MutationObserver(position);
    observer.observe(document.body, {subtree: true, childList: true, attributes: true, attributeFilter: ["hidden", "open"]});
    position();
  }
  function dismissTransient() {
    clearTimeout(timer);
    timer = setTimeout(() => {
      if (closeActiveHelp === close && !pinned && document.activeElement !== button &&
          !button.matches(":hover") && !note.matches(":hover")) close();
    }, 100);
  }
  button.addEventListener("focus", show);
  button.addEventListener("blur", dismissTransient);
  button.addEventListener("pointerenter", event => {
    if (event.pointerType !== "touch") show();
  });
  button.addEventListener("pointerleave", dismissTransient);
  note.addEventListener("pointerenter", () => clearTimeout(timer));
  note.addEventListener("pointerleave", dismissTransient);
  button.addEventListener("click", () => {
    if (pinned) close();
    else { show(); pinned = true; }
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
