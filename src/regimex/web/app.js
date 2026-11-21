import {$, notice} from "./ui.js";
import {createSession} from "./session.js";
import {createModels} from "./models.js";
import {createHistory} from "./history.js";
import {createComparison} from "./comparison.js";
import {createAnalysis} from "./analysis.js";
import {createIntegrations} from "./integrations.js";
import {createStorage} from "./storage.js";
import {createDeletion} from "./deletion.js";
import {createTraining} from "./training.js";

const session = createSession({
  onActivate: async () => {
    void storage.load();
    void deletion.load();
    void models.load();
    await Promise.all([history.load(), restoreLocation()]);
  },
  onReset: () => {
    screen = null;
    lastLocation = null;
    integrations.reset();
    training.reset();
    storage.reset();
    deletion.reset();
    history.reset();
    models.reset();
    analysis.reset();
  },
});
const getSelection = () => analysis.current;
const models = createModels({session, getSelection,
  getBusy: () => analysis.busy, getUploadReady: () => analysis.uploadReady,
  getBatchLocked: () => analysis.batchLocked});
const history = createHistory({session, getSelection, openAnalysis: (id) => analysis.open(id),
  deleteExperiment: (id, title) => deletion.open("experiment", id, title),
  compareAnalyses: async (first, second) => {
    const selection = await analysis.open(first.analysis_id);
    comparison.show(selection, second);
  }});
const comparison = createComparison({session, getSelection, rememberComparison});
const storage = createStorage({session});
const analysis = createAnalysis({session, models, history, comparison,
  onStorageChange: () => { void storage.load(); },
  onSelectionChange: () => deletion.close(),
  rememberAnalysis,
  deleteAnalysis: (id, title) => deletion.open("analysis", id, title)});
const deletion = createDeletion({session, onApplied: (target) => {
  analysis.forgetDeleted(target);
  void history.refreshAfterDeletion();
  void storage.load();
}});
const integrations = createIntegrations({session, navigate});
const training = createTraining({session, navigate, onPublication: () => models.load()});

const uuid = /^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$/i;
let screen = null, lastLocation = null;

function navigate(hash, replace = false) {
  const url = new URL(location.href);
  url.hash = hash;
  if (url.href !== location.href)
    window.history[replace ? "replaceState" : "pushState"](null, "", url);
  return restoreLocation();
}

function saveReportLocation(url, replace) {
  if (url.href === location.href) return;
  window.history[replace ? "replaceState" : "pushState"](null, "", url);
  lastLocation = url.href;
  $("open-analyses").href = url.href;
}

function rememberAnalysis(id) {
  if (screen !== "analyses") return;
  const url = new URL(location.href);
  if (id) url.searchParams.set("analysis", id);
  else url.searchParams.delete("analysis");
  url.searchParams.delete("compare");
  saveReportLocation(url, !id);
}

function rememberComparison(id) {
  if (screen !== "analyses") return;
  const url = new URL(location.href);
  if (id) url.searchParams.set("compare", id);
  else url.searchParams.delete("compare");
  saveReportLocation(url, false);
}

async function restoreLocation() {
  if (!session.value || lastLocation === location.href) return;
  lastLocation = location.href;
  const url = new URL(location.href);
  const match = /^#(training|connections)(?:\/([^/]+))?$/.exec(url.hash);
  const admin = session.value.principal.role === "admin";
  if (match && !admin) {
    url.hash = "";
    window.history.replaceState(null, "", url);
    lastLocation = url.href;
    notice((match[1] === "training" ? "Training" : "Connections") + " is available to administrators.");
  }
  const section = admin && match ? match[1] : "analyses";
  const id = match?.[2] || null;
  const previous = screen;
  if (previous !== section) {
    if (previous === "analyses") deletion.close();
    if (previous === "training") training.close();
    if (previous === "connections") integrations.reset();
  }
  screen = section;
  document.title = {analyses: "Regimex — Market analysis", training: "Model training · Regimex",
    connections: "Connections · Regimex"}[section];
  $("workspace").hidden = section !== "analyses";
  for (const name of ["analyses", "training", "connections"]) {
    const link = $("open-" + name);
    link.hidden = name !== "analyses" && !admin;
    if (section === name) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  const analysesUrl = new URL(url);
  analysesUrl.hash = "";
  $("open-analyses").href = analysesUrl.href;
  if (section === "training" || section === "connections") {
    if (id && !uuid.test(id) && !(section === "training" && id === "new"))
      return navigate("#" + section, true);
    return section === "training" ? training.show(id) : integrations.show(id);
  }
  if (previous && previous !== section) $("open-analyses").focus();
  const analysisId = url.searchParams.get("analysis");
  if (analysisId && uuid.test(analysisId)) {
    if (analysis.current?.id !== analysisId) await analysis.open(analysisId, {restore: true});
    else if (analysis.current.loading) await analysis.current.loading;
    const active = () => session.value && screen === "analyses" && location.href === url.href;
    const other = url.searchParams.get("compare");
    const comparedId = other && uuid.test(other) ? other : null;
    if (active() && comparison.comparedId !== comparedId)
      await comparison.restore(comparedId, active);
  } else if (analysis.current?.id) {
    deletion.close();
    analysis.clearSelection();
  }
}

for (const [id, hash] of [["open-analyses", ""], ["open-training", "#training"], ["open-connections", "#connections"]]) {
  $(id).addEventListener("click", event => {
    if (event.button || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    if (!session.value || (hash && session.value.principal.role !== "admin")) return;
    void navigate(hash);
  });
}
window.addEventListener("popstate", restoreLocation);
window.addEventListener("hashchange", restoreLocation);
void session.start();
