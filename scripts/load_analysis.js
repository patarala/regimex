import { fail, sleep } from "k6";
import exec from "k6/execution";
import http from "k6/http";
import { Counter, Trend } from "k6/metrics";

const port = Number(__ENV.REGIMEX_LOAD_PORT || 8001);
if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error("Invalid test port");
const api = `http://127.0.0.1:${port}/api/v1`;
const model = __ENV.REGIMEX_LOAD_MODEL || "tcn-0.2.0";
const token = __ENV.REGIMEX_LOAD_TOKEN;
const exportsRaw = [42, 43, 44, 45].map(seed => open(`${__ENV.REGIMEX_LOAD_DIR}/market-${seed}.json`));
const jobs = Number(__ENV.REGIMEX_LOAD_JOBS);
const clients = Number(__ENV.REGIMEX_LOAD_CLIENTS);
const resultTimeoutMs = 180_000;

const intakeTime = new Trend("analysis_intake", true);
const resultTime = new Trend("analysis_to_result", true);
const verified = new Counter("analyses_verified");
const thresholds = { analyses_verified: [`count==${jobs}`], http_req_failed: ["rate==0"] };
if (__ENV.REGIMEX_LOAD_MAX_RESULT_MS) {
  const budget = Number(__ENV.REGIMEX_LOAD_MAX_RESULT_MS);
  if (!Number.isFinite(budget) || budget <= 0) throw new Error("Invalid result latency budget");
  thresholds.analysis_to_result = [`max<=${budget}`];
}

export const options = {
  scenarios: {
    analyses: { executor: "shared-iterations", vus: clients, iterations: jobs, maxDuration: "5m" },
  },
  thresholds,
  maxRedirects: 0,
  systemTags: ["name", "method", "status", "scenario", "error_code", "expected_response"],
};

function request(name, path, expected, body = null, key = null) {
  const headers = { Authorization: `Bearer ${token}`, "Content-Type": "application/json" };
  if (key) headers["Idempotency-Key"] = key;
  const response = http.request(body === null ? "GET" : "POST", api + path, body, {
    headers, tags: { name }, timeout: "15s",
  });
  if (response.status !== expected) fail(`${name} returned HTTP ${response.status}, expected ${expected}`);
  return response;
}

export function setup() {
  verified.add(0);
  return exportsRaw.map(raw => {
    const input = JSON.parse(raw);
    // Upload the original bytes so JSON number formatting preserves the export digest.
    const snapshot = request("save snapshot", "/snapshots", 201, raw).json();
    if (snapshot.content_sha256 !== input.content_sha256) fail("Saved snapshot differs from input");
    const path = `/analyses?snapshot_id=${snapshot.snapshot_id}&limit=1`;
    const history = request("snapshot history", path, 200).json();
    return {
      id: snapshot.snapshot_id, digest: input.content_sha256, grid: input.time_grid,
      instruments: input.instruments.map(instrument => instrument.instrument_id),
      lastCreatedAt: history.items.length ? Date.parse(history.items[0].job.created_at) : null,
    };
  });
}

function checkResult(result, id, snapshot, resolved) {
  if (result.analysis_id !== id || result.snapshot_id !== snapshot.id ||
    result.snapshot_sha256 !== snapshot.digest || result.model.model_version_id !== model ||
    result.model.bundle_sha256 !== resolved.bundle_sha256 ||
    result.model.inference_policy_sha256 !== resolved.inference_policy_sha256) {
    fail("Result belongs to a different submission or model");
  }
  const gridMatches = Object.entries(snapshot.grid).every(([key, value]) => result.time_grid[key] === value);
  const instrumentsMatch = result.sequences.length === snapshot.instruments.length &&
    result.sequences.every((sequence, index) => sequence.instrument_id === snapshot.instruments[index] &&
      sequence.probabilities.length === snapshot.grid.step_count);
  if (!gridMatches || !instrumentsMatch) fail("Result differs from the input instrument or time grid");
}

export default function (snapshots) {
  const snapshot = snapshots[exec.scenario.iterationInTest % snapshots.length];
  const body = JSON.stringify({ snapshot_id: snapshot.id, model_version_id: model });
  const key = crypto.randomUUID();
  const started = Date.now();
  const accepted = request("submit analysis", "/analyses", 202, body, key);
  intakeTime.add(Date.now() - started);
  let analysis = accepted.json();
  const id = analysis.analysis_id;
  const resolved = analysis.job.model;
  if (!id || analysis.snapshot_id !== snapshot.id || resolved.model_version_id !== model) {
    fail("Accepted analysis differs from submission");
  }
  const created = Date.parse(analysis.job.created_at);
  if (!Number.isFinite(created) || (snapshot.lastCreatedAt !== null && created <= snapshot.lastCreatedAt)) {
    fail("Submission returned an analysis that predates this phase");
  }
  const path = `/analyses/${encodeURIComponent(id)}`;
  while (analysis.job.state !== "succeeded") {
    if (!["queued", "running"].includes(analysis.job.state)) {
      fail(`Analysis ${id}: ${analysis.job.state} (${analysis.job.failure?.code || "no failure code"})`);
    }
    if (Date.now() - started >= resultTimeoutMs) fail(`Analysis ${id} timed out`);
    sleep(0.5);
    analysis = request("poll analysis", path, 200).json();
    if (analysis.analysis_id !== id) fail("Status belongs to a different analysis");
  }
  if (!analysis.result_available) fail("Successful analysis has no result");
  const response = request("download result", path + "/result", 200);
  const elapsed = Date.now() - started;
  if (elapsed >= resultTimeoutMs) fail(`Analysis ${id} timed out`);
  checkResult(response.json(), id, snapshot, resolved);
  resultTime.add(elapsed);
  verified.add(1);
}
