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
  onActivate: restoreLocation,
  onReset: () => {
    analysesOpened = false;
    refreshHistoryOnOpen = false;
    screen = null;
    lastLocation = null;
    $("open-analyses").href = "/";
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
  chooseExperiment: experiment => analysis.chooseExperiment(experiment),
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
  onResultReady: () => restoreComparison(new URL(location.href)),
  rememberAnalysis,
  deleteAnalysis: (id, title) => deletion.open("analysis", id, title)});
const deletion = createDeletion({session, onApplied: (target, receipt) => {
  if (target.kind === "model_weights") {
    training.weightsRemoved(target.id, receipt.deletion_id);
  } else {
    analysis.forgetDeleted(target);
    void history.refreshAfterDeletion();
  }
  void storage.load();
}});
const integrations = createIntegrations({session, navigate});
const training = createTraining({session, navigate, deletion, onPublication: () => models.load()});

const uuid = /^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$/i;
let screen = null, lastLocation = null, analysesOpened = false;
let refreshHistoryOnOpen = false;

function navigate(path, replace = false) {
  const url = new URL(path, location.href);
  if (url.href !== location.href) {
    const state = url.pathname === "/" ? null : reportTabState(new URL($("open-analyses").href, url));
    window.history[replace ? "replaceState" : "pushState"](state, "", url);
  }
  return restoreLocation();
}

// Keep the Analyses tab on its selected report after a reload without putting
// those report IDs in a training or connection link someone might share.
function reportTabState(home) {
  const analysis = home.searchParams.get("analysis"), compare = home.searchParams.get("compare");
  if (!uuid.test(analysis)) return null;
  const search = new URLSearchParams({analysis});
  if (uuid.test(compare)) search.set("compare", compare);
  return {reportQuery: search.toString(), owner: session.value?.principal.owner_id ?? null};
}

// Bookmarks from the fragment router may also contain the last open analysis.
function restoreBookmark(url) {
  const match = url.pathname === "/" && /^#(training|connections)(?:\/([^/?]+))?(?:\?(.*))?$/.exec(url.hash);
  if (!match) return url;
  const [, section, id, query] = match;
  const target = new URL("/" + section, url);
  if (id && (uuid.test(id) || (section === "training" && id === "new"))) target.pathname += "/" + id;
  const view = new URLSearchParams(query).get("view");
  if (section === "training" && ["archived", "all"].includes(view)) target.searchParams.set("view", view);
  window.history.replaceState(reportTabState(url), "", target);
  return target;
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

function rememberComparison(id, replace = false) {
  if (screen !== "analyses") return;
  const url = new URL(location.href);
  if (id) url.searchParams.set("compare", id);
  else url.searchParams.delete("compare");
  saveReportLocation(url, replace);
}

function restoreComparison(url) {
  const active = () => session.value && screen === "analyses" && !document.hidden && location.href === url.href &&
    analysis.current?.id === url.searchParams.get("analysis");
  const other = url.searchParams.get("compare");
  const id = other && uuid.test(other) ? other : null;
  if (active()) return comparison.restore(id, active);
}

async function restoreLocation() {
  let url = restoreBookmark(new URL(location.href));
  if (!session.value || lastLocation === url.href) return;
  const match = /^\/(training|connections)(?:\/([^/]+))?$/.exec(url.pathname);
  const admin = session.value.principal.role === "admin";
  if (match) {
    const home = new URL("/", url), state = window.history.state;
    if (admin && state?.reportQuery && (state.owner === null || state.owner === session.value.principal.owner_id)) {
      home.search = state.reportQuery;
      if (state.owner === null) window.history.replaceState(reportTabState(home), "", url);
    }
    $("open-analyses").href = home.href;
    if (!admin) {
      url = home;
      window.history.replaceState(null, "", url);
      notice((match[1] === "training" ? "Training" : "Connections") + " is available to administrators.");
    }
  }
  lastLocation = url.href;
  const section = admin && match ? match[1] : "analyses";
  const id = match?.[2] || null;
  const previous = screen;
  if (previous !== section) {
    if (previous === "analyses" || previous === "training") deletion.close();
    if (previous === "analyses") comparison.cancelRestore();
    if (previous === "training") training.close();
    if (previous === "connections") integrations.reset();
  }
  screen = section;
  const resumed = analysis.setVisible(section === "analyses" && !document.hidden, url.searchParams.get("analysis"));
  const heading = $(section + "-title");
  document.title = heading.textContent + " · Regimex";
  $("workspace").hidden = section !== "analyses";
  for (const name of ["analyses", "training", "connections"]) {
    const link = $("open-" + name);
    link.hidden = name !== "analyses" && !admin;
    if (section === name) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  if (section === "training" || section === "connections") {
    if (section === "connections") return integrations.show(id);
    const view = url.searchParams.get("view");
    const pair = ["left", "right"].map(key => url.searchParams.get(key)).filter(id => uuid.test(id));
    return training.show(id, ["archived", "all"].includes(view) ? view : "active", pair);
  }
  let initialHistory;
  if (!analysesOpened) {
    analysesOpened = true;
    void storage.load();
    void deletion.load();
    void models.load();
    initialHistory = history.load();
  } else if (refreshHistoryOnOpen) {
    refreshHistoryOnOpen = false;
    initialHistory = history.refresh();
  }
  $("open-analyses").href = url.href;
  if (previous && previous !== section) heading.focus();
  await Promise.all([initialHistory, restoreAnalysis(url, resumed)]);
}

async function restoreAnalysis(url, resumed) {
  const analysisId = url.searchParams.get("analysis");
  if (analysisId && uuid.test(analysisId)) {
    if (analysis.current?.id !== analysisId) return analysis.open(analysisId, {restore: true});
    if (resumed) await resumed;
    else if (analysis.current.loading) await analysis.current.loading;
    if (analysis.current?.metadata) await restoreComparison(url);
  } else if (analysis.current) {
    deletion.close();
    analysis.clearSelection();
  }
}

window.addEventListener("click", event => {
  if (event.defaultPrevented || event.button || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
  const link = event.target.closest?.("a[data-workspace-link]");
  if (!link || !session.value || link.target || link.hasAttribute("download")) return;
  event.preventDefault();
  void navigate(link.href);
});
window.addEventListener("popstate", restoreLocation);
window.addEventListener("hashchange", restoreLocation);
// Returning from a terminal can focus an already-visible window or reveal a hidden tab.
let returnTimer;
function refreshOnReturn() {
  clearTimeout(returnTimer);
  if (document.hidden || !screen) return;
  returnTimer = setTimeout(() => {
    if (screen === "analyses") void history.refresh();
    else refreshHistoryOnOpen = analysesOpened;
    if (screen === "training") training.refreshAfterReturn();
  }, 80);
}
window.addEventListener("focus", refreshOnReturn);
document.addEventListener("visibilitychange", () => {
  refreshOnReturn();
  const visible = screen === "analyses" && !document.hidden;
  const resumed = analysis.setVisible(visible);
  if (!visible) comparison.cancelRestore();
  else if (!resumed && analysis.current?.metadata?.result_available) void restoreComparison(new URL(location.href));
});
restoreBookmark(new URL(location.href));
void session.start();
