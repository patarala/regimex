import {createSession} from "./session.js";
import {createModels} from "./models.js";
import {createHistory} from "./history.js";
import {createComparison} from "./comparison.js";
import {createAnalysis} from "./analysis.js";
import {createIntegrations} from "./integrations.js";
import {createStorage} from "./storage.js";
import {createDeletion} from "./deletion.js";

const session = createSession({
  onActivate: async () => {
    integrations.activate();
    void storage.load();
    void deletion.load();
    void models.load();
    const selected = new URL(location.href).searchParams.get("analysis");
    await Promise.all([
      history.load(),
      selected && /^[0-9a-f-]{36}$/.test(selected) ? analysis.open(selected) : undefined,
    ]);
  },
  onReset: () => {
    integrations.reset();
    storage.reset();
    deletion.reset();
    history.reset();
    models.reset();
    analysis.reset();
  },
});
const getSelection = () => analysis.current;
const models = createModels({session, getSelection,
  getBusy: () => analysis.busy, getExport: () => analysis.exportText});
const history = createHistory({session, getSelection, openAnalysis: (id) => analysis.open(id),
  deleteExperiment: (id, title) => deletion.open("experiment", id, title),
  compareAnalyses: async (first, second) => {
    const selection = await analysis.open(first.analysis_id);
    comparison.show(selection, second);
  }});
const comparison = createComparison({session, getSelection});
const storage = createStorage({session});
const analysis = createAnalysis({session, models, history, comparison,
  onStorageChange: () => { void storage.load(); },
  onSelectionChange: () => deletion.close(),
  deleteAnalysis: (id, title) => deletion.open("analysis", id, title)});
const deletion = createDeletion({session, onApplied: (target) => {
  analysis.forgetDeleted(target);
  void history.refreshAfterDeletion();
  void storage.load();
}});
const integrations = createIntegrations({session});
void session.start();
