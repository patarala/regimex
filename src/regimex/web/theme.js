(() => {
  const key = "regimex-theme";
  const root = document.documentElement;

  function apply(value) {
    if (value === "light" || value === "dark") root.dataset.theme = value;
    else delete root.dataset.theme;
  }

  function restore() {
    try { apply(localStorage.getItem(key)); }
    catch { apply("system"); }
  }

  // Apply the saved scheme before the stylesheet paints the page.
  restore();
  document.addEventListener("DOMContentLoaded", () => {
    const toggle = document.getElementById("theme");
    const media = matchMedia("(prefers-color-scheme: dark)");
    const previews = [document.getElementById("report-preview"), document.getElementById("comparison-preview")];
    const systemTheme = () => media.matches ? "dark" : "light";
    const currentTheme = () => root.dataset.theme || systemTheme();
    function sendTheme(frame) {
      if (!frame.hasAttribute("src")) return;
      // Sandboxed reports have an opaque origin. Only send the appearance preference.
      frame.contentWindow?.postMessage({type: "regimex-theme", theme: currentTheme()}, "*");
    }
    function sync() {
      toggle.setAttribute("aria-pressed", String(currentTheme() === "dark"));
      previews.forEach(sendTheme);
    }
    for (const frame of previews) frame.addEventListener("load", () => sendTheme(frame));
    media.addEventListener("change", () => { if (!root.dataset.theme) sync(); });
    sync();
    toggle.hidden = false;
    toggle.addEventListener("click", () => {
      const next = currentTheme() === "dark" ? "light" : "dark";
      apply(next === systemTheme() ? "system" : next);
      sync();
      try {
        if (root.dataset.theme) localStorage.setItem(key, root.dataset.theme);
        else localStorage.removeItem(key);
      } catch { /* The selection still applies in this tab when storage is disabled. */ }
    });
    window.addEventListener("storage", event => {
      if (event.key !== key && event.key !== null) return;
      restore();
      sync();
    });
  });
})();
