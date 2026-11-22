import {$, analysisStateLabel, element, shortDate} from "./ui.js";
import {architectureName} from "./training-form.js";
import {differentSettings} from "./training-json.js";

function table(caption, links, rows) {
  const result = element("table");
  result.append(element("caption", caption, "visually-hidden"));
  const heading = result.createTHead().insertRow();
  for (const value of ["", ...links]) {
    const cell = element("th");
    cell.scope = "col";
    cell.append(value);
    heading.append(cell);
  }
  const body = result.createTBody();
  for (const [label, left, right] of rows) {
    const row = body.insertRow(), name = element("th", label);
    name.scope = "row";
    row.append(name, element("td", left), element("td", right));
  }
  return result;
}

export function createTrainingComparison({readRun, session, runPath, duration}) {
  const panel = $("training-comparison"), status = $("training-comparison-status");
  const content = $("training-comparison-results"), retry = $("training-comparison-retry");
  let generation = 0, pair = [];

  function close() {
    generation++;
    pair = [];
    panel.hidden = true;
    content.replaceChildren();
    status.textContent = "";
    retry.hidden = true;
  }

  async function show(ids, focus = true) {
    const stamp = ++generation;
    pair = ids;
    panel.hidden = false;
    content.replaceChildren();
    retry.hidden = true;
    if (focus) $("training-comparison-title").focus();
    if (ids.length !== 2 || ids[0] === ids[1]) {
      status.textContent = "Select two different runs in the history and choose Compare.";
      return;
    }
    status.textContent = "Loading runs…";
    try {
      const runs = await Promise.all(ids.map(readRun));
      if (generation !== stamp) return;
      const names = runs.map(run => run.display_name || architectureName(run.training.architecture_id, true));
      const links = () => runs.map((run, index) => {
        const suffix = names[0] === names[1] ? " #" + run.run_id.slice(-8) : "";
        const link = element("a", (index === 0 ? "A · " : "B · ") + names[index] + suffix);
        link.href = runPath(run.run_id);
        link.dataset.workspaceLink = "";
        return link;
      });
      const row = (name, select) => [name, ...runs.map(run => select(run) ?? "—")];
      const rows = [
        row("Status", r => analysisStateLabel(r.job.state)),
        row("Created", r => shortDate(r.job.created_at)),
        row("Dataset", r => r.training.dataset_id),
        row("Architecture", r => architectureName(r.training.architecture_id, true)),
        row("Trainable parameters", r => r.trainable_parameters?.toLocaleString()),
        ...[["Elapsed", "elapsed_seconds"], ["In queue", "queued_seconds"],
          ["With worker", "running_seconds"], ["Model fit time", "fit_seconds"]].map(([name, key]) =>
          row(name, r => r.statistics?.[key] == null ? null : duration(r.statistics[key]))),
      ];
      const overview = table("Run results", links(), rows);
      content.append(overview);
      const differences = differentSettings(runs[0].configuration, runs[1].configuration);
      content.append(element("h3", "Settings that differ"), differences.length
        ? table("Settings that differ", links(), differences)
        : element("p", "Both runs used the same settings.", "hint"));
      if (!runs.every(run => run.job.state === "succeeded")) {
        status.textContent = "Both runs must succeed to compare their validation scores.";
        return;
      }
      const [a, b] = runs.map(run => run.training);
      if (!["dataset_id", "validation_reference_set_id", "analysis_profile_id"].every(key =>
        a[key] && a[key] === b[key])) {
        status.textContent = "Comparing validation scores requires the same dataset, validation labels and analysis profile.";
        return;
      }
      status.textContent = "Loading validation scores…";
      const values = await Promise.all(runs.map(async run => {
        if (run.output?.validation_segment_macro_f1 != null) return run.output.validation_segment_macro_f1;
        const score = await session.json("/api/v1/training-runs/" + encodeURIComponent(run.run_id) + "/validation-score");
        return score.validation_segment_macro_f1;
      }));
      if (generation !== stamp) return;
      const scoreRow = overview.tBodies[0].insertRow(), label = element("th", "Validation interval macro-F1");
      label.scope = "row";
      scoreRow.append(label, ...values.map(value => element("td", value.toFixed(4))));
      const change = values[1] - values[0];
      status.textContent = "Validation interval macro-F1 (B − A): " + (change > 0 ? "+" : "") + change.toFixed(4);
    } catch (error) {
      if (generation !== stamp) return;
      status.textContent = error.message;
      retry.hidden = false;
    }
  }
  retry.addEventListener("click", () => show(pair, false));
  return {close, show};
}
