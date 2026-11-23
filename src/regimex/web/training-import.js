import {element} from "./ui.js";
import {bytes} from "./storage.js";

export const isImported = run => run.training.origin === "external";

export function importedFacts(run) {
  const {evidence, weights, model_version_id, build_id, implementation_version, analysis_profile_id} = run.training;
  return [
    ["Model version", model_version_id],
    ["Original run", evidence.run_id],
    ["Build", build_id],
    ["Implementation version", implementation_version],
    ["Analysis profile", analysis_profile_id],
    ["Bundle SHA-256", run.output?.bundle_sha256],
    ["Model weights", weights.path + " · " + bytes(weights.size_bytes) + (run.weights_deletion_id ? " · removed" : " · saved")],
    ["Weights SHA-256", weights.sha256],
    ["Training dataset SHA-256", evidence.dataset_sha256],
    ["Split manifest SHA-256", evidence.split_manifest_sha256],
    ["Validation dataset SHA-256", evidence.validation_dataset_sha256 || evidence.dataset_sha256],
    ["Training labels", evidence.labels.reference_set_id],
    ["Validation labels", evidence.validation_references.reference_set_id],
    ["Validation labels SHA-256", evidence.validation_references.content_sha256],
    ...(evidence.parent_models || []).map(model => ["Parent model", model.model_version_id]),
    ...(evidence.additional_training_sources || []).map(source => ["Additional training source", source.source_id]),
  ].filter(([, value]) => value != null);
}

function facts(entries) {
  const list = element("dl", "", "training-facts");
  for (const [label, value] of entries) {
    if (value != null) list.append(element("dt", label), element("dd", String(value)));
  }
  return list;
}

function latencyTable(latency) {
  const section = element("section");
  const wrapper = element("div", "", "training-values");
  const table = element("table");
  table.createCaption().textContent = "Median inference latency (ms)";
  const header = table.createTHead().insertRow();
  const hasPrediction = latency.timings.some(row => row.predict_ms != null);
  for (const label of ["Steps", ...(hasPrediction ? ["Prediction"] : []), "Prediction + decoding"]) {
    const cell = element("th", label);
    cell.scope = "col";
    header.append(cell);
  }
  const body = table.createTBody();
  for (const {steps, predict_ms, predict_decode_ms} of latency.timings) {
    const row = body.insertRow();
    row.insertCell().textContent = steps.toLocaleString();
    if (hasPrediction) row.insertCell().textContent = predict_ms ?? "—";
    row.insertCell().textContent = predict_decode_ms;
  }
  const details = element("details");
  details.append(element("summary", "Measurement provenance"), facts([
    ["Threads", latency.threads], ["Bundle SHA-256", latency.manifest_sha256],
    ["Report SHA-256", latency.report_sha256],
  ]));
  wrapper.append(table);
  section.append(element("h4", "Inference latency"), wrapper, details);
  return section;
}

export function showImportedResults(content, results, latency) {
  content.replaceChildren(...results.map(result => {
    const native = result.kind === "native_final";
    const item = element("section");
    const title = element("h4", native ? "Native final benchmark" : "Controlled transfer");
    const details = element("details");
    details.append(element("summary", "Evaluation provenance"), facts([
      ["Candidate", result.candidate], ["Bundle SHA-256", result.bundle_sha256],
      ["Report SHA-256", result.report_sha256], ["Freeze SHA-256", result.freeze_sha256],
      ["Protocol SHA-256", result.protocol_sha256],
    ]));
    item.append(title, facts([
      ["Interval macro-F1", result.interval_macro_f1],
      ["Per-step macro-F1", result.point_macro_f1],
      [native ? "Experiments" : "Paired paths", result.sample_count],
      ["Quality gate", result.passed ? "Passed" : "Failed"],
    ]), details);
    return item;
  }));
  if (latency) content.append(latencyTable(latency));
}
