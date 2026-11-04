import {createSession} from "./session.js";
import {createModels} from "./models.js";
import {createHistory} from "./history.js";
import {createComparison} from "./comparison.js";
import {createAnalysis} from "./analysis.js";

const session = createSession({
  onActivate: async () => {
    void models.load();
    await history.load();
    const selected = new URL(location.href).searchParams.get("analysis");
    if (selected && /^[0-9a-f-]{36}$/.test(selected)) await analysis.open(selected);
  },
  onReset: () => {
    history.reset();
    models.reset();
    analysis.reset();
  },
});
const getSelection = () => analysis.current;
const models = createModels({session, getSelection,
  getBusy: () => analysis.busy, getExport: () => analysis.exportText});
const history = createHistory({session, getSelection, openAnalysis: (id) => analysis.open(id)});
const comparison = createComparison({session, getSelection});
const analysis = createAnalysis({session, models, history, comparison});
void session.start();
