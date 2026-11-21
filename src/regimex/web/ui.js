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
let dismissedHoverAt;

function resumeHoverOnMovement(event) {
  if (event.clientX === dismissedHoverAt.x && event.clientY === dismissedHoverAt.y) return false;
  dismissedHoverAt = undefined;
  document.removeEventListener("pointermove", resumeHoverOnMovement);
  return true;
}

function bindHover(control, enter) {
  function hover(event) {
    if (event.pointerType === "touch") return;
    if (dismissedHoverAt) {
      // Escape can uncover a different control without the pointer moving.
      if (!resumeHoverOnMovement(event)) return;
    } else if (event.type === "pointermove") return;
    enter();
  }
  control.addEventListener("pointerenter", hover);
  control.addEventListener("pointermove", hover);
}

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
  note.style.maxWidth = `min(var(--tooltip-max-width, 22rem), ${width - 2 * margin}px)`;
  note.style.maxHeight = Math.max(above, below, 0) + "px";
  const panel = note.getBoundingClientRect();
  const centered = anchor.left + (anchor.width - panel.width) / 2;
  note.style.left = Math.max(left + margin, Math.min(centered, left + width - margin - panel.width)) + "px";
  // Adjoining edges let the pointer move straight from the button onto the note.
  note.style.top = (below >= panel.height || below >= above ? anchor.bottom : anchor.top - panel.height) + "px";
  return true;
}

export function bindTooltip(button, note, {field, trigger} = {}) {
  note.setAttribute("role", "tooltip");
  note.setAttribute("popover", "manual");
  note.hidden = true;
  button.setAttribute("aria-controls", note.id);
  button.setAttribute("aria-expanded", "false");
  for (const control of [button, field, trigger].filter(Boolean)) {
    const descriptions = control.getAttribute("aria-describedby");
    control.setAttribute("aria-describedby", descriptions ? descriptions + " " + note.id : note.id);
  }

  let pinned = false, timer, observer, visibility, listeners, pointerOnNote;
  function close() {
    clearTimeout(timer);
    if (closeActiveHelp !== close) return;
    observer.disconnect();
    visibility?.disconnect();
    listeners.abort();
    if (note.hidePopover && note.matches(":popover-open")) note.hidePopover();
    note.hidden = true;
    pinned = false;
    button.setAttribute("aria-expanded", "false");
    closeActiveHelp = undefined;
  }
  function position() {
    if (!note.isConnected || !positionHelp(button, note)) close();
  }
  function show() {
    clearTimeout(timer);
    if (closeActiveHelp === close || !note.isConnected || !button.getClientRects().length) return;
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
      if (pointerOnNote && note.matches(":hover")) {
        dismissedHoverAt = pointerOnNote;
        document.addEventListener("pointermove", resumeHoverOnMovement);
      }
      if (document.activeElement === field) button.focus({preventScroll: true});
      close();
      event.stopPropagation();
    }, {capture: true, signal});
    document.addEventListener("focusin", event => {
      if (pinned && event.target !== field && event.target !== trigger && event.target !== button &&
          !note.contains(event.target)) close();
    }, {signal});
    document.addEventListener("scroll", position, {capture: true, passive: true, signal});
    window.addEventListener("resize", position, {signal});
    window.visualViewport?.addEventListener("resize", position, {signal});
    window.visualViewport?.addEventListener("scroll", position, {signal});
    window.addEventListener("hashchange", close, {signal});
    window.addEventListener("pagehide", close, {signal});
    observer = new MutationObserver(position);
    observer.observe(document.body, {subtree: true, childList: true, attributes: true, attributeFilter: ["hidden", "open"]});
    if (trigger) {
      // A top-layer note can remain visible after its row scrolls out of a list.
      visibility = new IntersectionObserver(([entry]) => {
        if (entry.intersectionRatio < 0.99) close();
      }, {threshold: 0.99});
      visibility.observe(button);
    }
    position();
  }
  function dismissTransient() {
    clearTimeout(timer);
    timer = setTimeout(() => {
      if (closeActiveHelp === close && !pinned && document.activeElement !== button &&
          !button.matches(":hover") && !note.matches(":hover") && !trigger?.matches(":hover, :focus-visible")) close();
    }, 100);
  }
  button.addEventListener("focus", show);
  button.addEventListener("blur", dismissTransient);
  bindHover(button, show);
  button.addEventListener("pointerleave", dismissTransient);
  function trackNotePointer(event) {
    pointerOnNote = {x: event.clientX, y: event.clientY};
    clearTimeout(timer);
  }
  note.addEventListener("pointerenter", trackNotePointer);
  note.addEventListener("pointermove", trackNotePointer);
  note.addEventListener("pointerleave", dismissTransient);
  button.addEventListener("click", () => {
    if (pinned) close();
    else { show(); pinned = true; }
  });
  if (trigger) {
    trigger.addEventListener("focus", () => { if (trigger.matches(":focus-visible")) show(); });
    trigger.addEventListener("blur", dismissTransient);
    bindHover(trigger, () => {
      clearTimeout(timer);
      timer = setTimeout(() => { if (trigger.matches(":hover")) show(); }, 300);
    });
    trigger.addEventListener("pointerleave", dismissTransient);
    trigger.addEventListener("click", close);
  }
}

export function addFieldHelp(label, field, text) {
  const heading = element("div", "", "field-heading");
  const note = element("p", text, "field-help");
  note.id = field.id + "-help";
  const button = element("button", "", "field-help-toggle");
  button.type = "button";
  const icon = element("span", "?");
  icon.setAttribute("aria-hidden", "true");
  button.append(icon, element("span", "Help for " + label.textContent, "visually-hidden"));
  label.replaceWith(heading);
  heading.append(label, button);
  field.after(note);
  bindTooltip(button, note, {field});
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
