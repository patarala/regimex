import { fail } from "k6";

export function integer(name, fallback, max, min = 1) {
  const value = Number(__ENV[name] || fallback);
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new Error(`${name} must be an integer between ${min} and ${max}`);
  }
  return value;
}

export function loadTarget() {
  const scheme = __ENV.REGIMEX_LOAD_SCHEME || "http";
  if (scheme !== "http" && scheme !== "https") throw new Error("REGIMEX_LOAD_SCHEME must be http or https");
  const port = integer("REGIMEX_LOAD_PORT", scheme === "https" ? 8443 : 8001, 65535);
  const host = __ENV.REGIMEX_LOAD_HOST || (scheme === "https" ? "localhost" : "127.0.0.1");
  const labels = host.split(".");
  if (host.length > 253 || !labels.every(label => /^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$/.test(label))) {
    throw new Error("REGIMEX_LOAD_HOST must be a hostname without credentials, port or path");
  }
  if (scheme === "http" && !["127.0.0.1", "localhost"].includes(host)) {
    throw new Error("A remote load target requires HTTPS");
  }
  return `${scheme}://${host}:${port}/api/v1`;
}

const classNames = ["sideways", "growth", "decline"];

export function validDigest(value) {
  return typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
}

function sameMetadata(expected, actual) {
  if (expected === actual) return true;
  if (expected === null || actual === null || typeof expected !== "object" || typeof actual !== "object" ||
    Array.isArray(expected) !== Array.isArray(actual)) return false;
  const keys = Object.keys(expected);
  return keys.length === Object.keys(actual).length && keys.every(key => sameMetadata(expected[key], actual[key]));
}

function checkSequence(sequence, instrument, steps) {
  if (sequence.instrument_id !== instrument || !Array.isArray(sequence.probabilities) ||
    sequence.probabilities.length !== steps) {
    fail("Result differs from the input instrument or time grid");
  }
  for (const row of sequence.probabilities) {
    if (!Array.isArray(row) || row.length !== 3 ||
      !row.every(p => Number.isFinite(p) && p >= 0 && p <= 1) ||
      Math.abs(row[0] + row[1] + row[2] - 1) > 1e-6) fail("Invalid direction probabilities");
  }
  if (!Array.isArray(sequence.intervals)) fail("Missing direction intervals");
  let stop = 0;
  for (const interval of sequence.intervals) {
    if (interval.instrument_id !== instrument || interval.start !== stop ||
      !Number.isInteger(interval.stop) || interval.stop <= stop || interval.stop > steps) {
      fail("Invalid direction interval coverage");
    }
    if (interval.status === "observed") {
      if (!classNames.includes(interval.direction)) fail("Invalid observed interval direction");
    } else if (!["no_trades", "insufficient_data"].includes(interval.status) || interval.direction !== null) {
      fail("Invalid unavailable interval classification");
    }
    stop = interval.stop;
  }
  if (stop !== steps) fail("Direction intervals do not cover the complete grid");
}

export function checkResult(result, id, snapshot, resolved, model) {
  if (result.result_version !== "2.0" || result.analysis_id !== id ||
    result.snapshot_id !== snapshot.id || result.snapshot_sha256 !== snapshot.digest ||
    result.model.model_version_id !== model ||
    result.model.bundle_sha256 !== resolved.bundle_sha256 ||
    result.model.inference_policy_sha256 !== resolved.inference_policy_sha256) {
    fail("Result belongs to a different submission or model");
  }
  if (!Array.isArray(result.class_names) || result.class_names.length !== classNames.length ||
    !classNames.every((name, index) => name === result.class_names[index])) {
    fail("Invalid direction classes");
  }
  if (!sameMetadata(snapshot.source, result.source)) fail("Result source differs from input");
  if (!sameMetadata(snapshot.grid, result.time_grid) || !sameMetadata(snapshot.instruments, result.instruments) ||
    !Array.isArray(result.sequences) || result.sequences.length !== snapshot.instruments.length) {
    fail("Result differs from the input instruments or time grid");
  }
  snapshot.instruments.forEach((instrument, index) => {
    checkSequence(result.sequences[index], instrument.instrument_id, snapshot.grid.step_count);
  });
}

export function rejectionCode(response) {
  try {
    const code = response.json("error.code");
    if (typeof code === "string" && /^[a-z0-9_]{1,64}$/.test(code)) return code;
  } catch {
    // A transport failure or a non-JSON error page can have no API error code.
  }
  return "unknown";
}

export function checkLaunchSource(job) {
  if (job?.launch_source?.channel !== "api" || job.launch_source.command !== null) {
    fail("Direct API submission has an unexpected launch source");
  }
}

export function checkStatus(view, { id, operation, snapshot, pinned }) {
  if (view.analysis_id !== id || view.job?.operation_id !== operation ||
      view.snapshot_id !== snapshot.id || view.snapshot_sha256 !== snapshot.digest ||
      !sameMetadata(pinned, view.job.model)) {
    fail("Status differs from the accepted analysis, operation, snapshot or model");
  }
  checkLaunchSource(view.job);
}
