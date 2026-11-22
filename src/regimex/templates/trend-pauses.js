"use strict";

const trendPausePercent = new Intl.NumberFormat("en", {
  maximumSignificantDigits: 3, signDisplay: "exceptZero",
});

function trendPauseDescription(pause) {
  const duration = pause.stop - pause.start;
  const change = pause.price_change_percent === null
    ? "unavailable" : trendPausePercent.format(pause.price_change_percent) + "%";
  return `Pause in ${pause.direction} · ${duration} ${duration === 1 ? "step" : "steps"} · price change ${change}`;
}

function trendPauseOverlay(pauses, stepCount, boundary, color) {
  if (!pauses.length) return {shapes: [], notes: undefined};
  const notes = Array(stepCount).fill("");
  const shapes = pauses.map((pause) => {
    notes.fill("<br>" + trendPauseDescription(pause), pause.start, pause.stop);
    return {
      type: "rect", xref: "x", yref: "y domain",
      x0: boundary(pause.start), x1: boundary(pause.stop), y0: 0, y1: 1,
      fillcolor: color, opacity: 0.22, line: {width: 0}, layer: "below",
    };
  });
  return {shapes, notes};
}

function renderTrendPauseList(pauses, enabled, instrumentId, onZoom) {
  const list = document.getElementById("pause-list");
  list.replaceChildren();
  for (const pause of pauses) {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "interval-link";
    button.textContent = `Steps [${pause.start}, ${pause.stop})`;
    button.setAttribute("aria-label", `Zoom to ${instrumentId} pause at steps [${pause.start}, ${pause.stop})`);
    button.addEventListener("click", () => onZoom(pause));
    item.append(button, " · " + trendPauseDescription(pause));
    list.append(item);
  }
  document.getElementById("pause-details").hidden = !pauses.length;
  document.getElementById("pause-count").textContent = pauses.length === 1
    ? "1 highlighted price pause" : pauses.length + " highlighted price pauses";
  document.getElementById("pause-empty").hidden = !enabled || pauses.length > 0;
}
