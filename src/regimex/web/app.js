import {createSession} from "./session.js";
import {createModels} from "./models.js";
import {createHistory} from "./history.js";
import {createComparison} from "./comparison.js";
import {createAnalysis} from "./analysis.js";
import {createIntegrations} from "./integrations.js";
import {createStorage} from "./storage.js";

const session = createSession({
  onActivate: async () => {
    integrations.activate();
    void storage.load();
    void models.load();
    await history.load();
    const selected = new URL(location.href).searchParams.get("analysis");
    if (selected && /^[0-9a-f-]{36}$/.test(selected)) await analysis.open(selected);
  },
  onReset: () => {
    integrations.reset();
    storage.reset();
    history.reset();
    models.reset();
    analysis.reset();
  },
});
const getSelection = () => analysis.current;
const models = createModels({session, getSelection,
  getBusy: () => analysis.busy, getExport: () => analysis.exportText});
const history = createHistory({session, getSelection, openAnalysis: (id) => analysis.open(id),
  compareAnalyses: async (first, second) => {
    const selection = await analysis.open(first.analysis_id);
    comparison.show(selection, second);
  }});
const comparison = createComparison({session, getSelection});
const storage = createStorage({session});
const analysis = createAnalysis({session, models, history, comparison,
  onStorageChange: () => { void storage.load(); }});
const integrations = createIntegrations({session});
void session.start();
