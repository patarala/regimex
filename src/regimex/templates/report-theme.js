"use strict";

const reportMedia = matchMedia("(prefers-color-scheme: dark)");

function reportIsDark() {
  const explicit = document.documentElement.dataset.theme;
  return explicit ? explicit === "dark" : reportMedia.matches;
}

function watchReportTheme(redraw = () => {}) {
  let dark = reportIsDark();
  function update() {
    const next = reportIsDark();
    if (dark !== next) {
      dark = next;
      redraw();
    }
  }
  reportMedia.addEventListener("change", update);
  if (window.parent !== window) {
    // A sandboxed preview has an opaque origin. Only its embedding window sets its appearance.
    window.addEventListener("message", event => {
      if (event.source !== window.parent || event.data?.type !== "regimex-theme") return;
      const theme = event.data.theme;
      if (theme !== "light" && theme !== "dark") return;
      document.documentElement.dataset.theme = theme;
      update();
    });
  }
}
