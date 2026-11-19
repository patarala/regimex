import { fail, sleep } from "k6";
import exec from "k6/execution";
import http from "k6/http";
import { Counter, Trend } from "k6/metrics";

function integer(name, fallback, max, min = 1) {
  const value = Number(__ENV[name] || fallback);
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new Error(`${name} must be an integer between ${min} and ${max}`);
  }
  return value;
}

const port = integer("REGIMEX_LOAD_PORT", 8001, 65535);
const api = `http://127.0.0.1:${port}/api/v1`;
const model = __ENV.REGIMEX_LOAD_MODEL || "tcn-0.2.0";
const expectedBundle = __ENV.REGIMEX_LOAD_EXPECTED_BUNDLE_SHA256;
const expectedPolicy = __ENV.REGIMEX_LOAD_EXPECTED_POLICY_SHA256;
for (const digest of [expectedBundle, expectedPolicy]) {
  if (digest !== undefined && !validDigest(digest)) throw new Error("Invalid expected model digest");
}
const token = __ENV.REGIMEX_LOAD_TOKEN;
const exportsRaw = [42, 43, 44, 45].map(seed => open(`${__ENV.REGIMEX_LOAD_DIR}/market-${seed}.json`));
const phase = __ENV.REGIMEX_LOAD_PHASE || "analysis";
const sustained = phase === "sustained";
const resultTimeoutMs = 180_000;
const drainTime = "3m30s";
const rate = sustained && integer("REGIMEX_LOAD_RATE_PER_MINUTE", 6, 120);
const seconds = sustained && integer("REGIMEX_LOAD_DURATION_SECONDS", 120, 600, 2);
const jobs = sustained ? rate * seconds / 60 : Number(__ENV.REGIMEX_LOAD_JOBS);
if (sustained && (!Number.isInteger(jobs) || jobs < 1 || jobs > 100)) {
  throw new Error("Sustained rate × seconds / 60 must request 1 to 100 whole arrivals");
}

const intakeTime = new Trend("analysis_intake", true);
const resultTime = new Trend("analysis_to_result", true);
const scheduled = new Counter("analyses_scheduled");
const started = new Counter("analyses_started");
const accepted = new Counter("analyses_accepted");
const verified = new Counter("analyses_verified");
const quotaRefused = new Counter("analyses_quota_refused");
const submitFailed = new Counter("analyses_submit_failed");
const failed = new Counter("analyses_failed");
const accounted = new Counter("analyses_accounted");
const queuedFirst = new Trend("analysis_queued_first_half");
const queuedSecond = new Trend("analysis_queued_second_half");
const firstSamples = new Counter("analysis_queue_first_half_samples");
const secondSamples = new Counter("analysis_queue_second_half_samples");
const activeAfterDrain = new Trend("analysis_active_after_drain");
const drainSamples = new Counter("analysis_after_drain_samples");
const sampleFailed = new Counter("analysis_queue_sample_errors");

const thresholds = {
  analyses_started: [`count==${jobs}`],
  analyses_accounted: [`count==${jobs}`],
  analyses_verified: [sustained ? "count>0" : `count==${jobs}`],
  analyses_failed: ["count==0"],
  http_req_failed: ["rate==0"],
};
if (sustained) {
  thresholds.dropped_iterations = ["count==0"];
  thresholds.analysis_queue_sample_errors = ["count==0"];
  thresholds.analysis_queue_first_half_samples = ["count>0"];
  thresholds.analysis_queue_second_half_samples = ["count>0"];
  thresholds.analysis_after_drain_samples = ["count==1"];
  thresholds.analysis_active_after_drain = ["max==0"];
}
if (__ENV.REGIMEX_LOAD_MAX_RESULT_MS) {
  const budget = Number(__ENV.REGIMEX_LOAD_MAX_RESULT_MS);
  if (!Number.isFinite(budget) || budget <= 0) throw new Error("Invalid result latency budget");
  thresholds.analysis_to_result = [`max<=${budget}`];
}

const scenarios = sustained ? {
  analyses: {
    // k6 schedules the first arrival at zero. End between ticks to exclude the window's right edge.
    executor: "constant-arrival-rate", rate, timeUnit: "1m",
    duration: `${Math.round(seconds * 1000 - 30_000 / rate)}ms`,
    preAllocatedVUs: integer("REGIMEX_LOAD_MAX_VUS", 16, 32), gracefulStop: drainTime,
  },
  queue: {
    executor: "constant-vus", exec: "sampleQueue", vus: 1, duration: `${seconds}s`,
    gracefulStop: "25s",
  },
} : {
  analyses: {
    executor: "shared-iterations", vus: Number(__ENV.REGIMEX_LOAD_CLIENTS),
    iterations: jobs, maxDuration: "5m",
  },
};

export const options = {
  scenarios, thresholds,
  maxRedirects: 0,
  systemTags: ["name", "method", "status", "scenario", "error_code", "expected_response"],
};

function send(name, path, body, statuses, key) {
  const headers = { Authorization: `Bearer ${token}`, "Content-Type": "application/json" };
  if (key) headers["Idempotency-Key"] = key;
  return http.request(body === null ? "GET" : "POST", api + path, body, {
    headers, tags: { name }, timeout: "15s", responseCallback: http.expectedStatuses(...statuses),
  });
}

function request(name, path, expected, body = null) {
  const response = send(name, path, body, [expected]);
  if (response.status !== expected) fail(`${name} returned HTTP ${response.status}, expected ${expected}`);
  return response;
}

function ownerCount(state) {
  const items = request(`count ${state} analyses`, `/analyses?state=${state}&limit=100`, 200).json("items");
  if (!Array.isArray(items)) fail(`Missing analysis list for state=${state}`);
  return items.length;
}

export function setup() {
  scheduled.add(jobs);
  for (const counter of [started, accepted, verified, quotaRefused, submitFailed, failed, accounted]) {
    counter.add(0);
  }
  if (sustained) {
    for (const counter of [sampleFailed, firstSamples, secondSamples, drainSamples]) counter.add(0);
    if (ownerCount("active") !== 0) fail("The sustained profile needs a dedicated idle test owner");
  }
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

export function sampleQueue() {
  try {
    const queued = ownerCount("queued");
    const elapsed = Date.now() - exec.scenario.startTime;
    // An in-flight poll can finish during the drain, beyond both intake windows.
    if (elapsed >= seconds * 1000) return;
    if (elapsed < seconds * 500) {
      queuedFirst.add(queued);
      firstSamples.add(1);
    } else {
      queuedSecond.add(queued);
      secondSamples.add(1);
    }
    console.log(`REGIMEX_QUEUE ${JSON.stringify({ elapsed_ms: elapsed, queued })}`);
  } catch (error) {
    sampleFailed.add(1);
    throw error;
  } finally {
    sleep(Math.min(5, seconds / 4));
  }
}

export function teardown() {
  if (sustained) {
    activeAfterDrain.add(ownerCount("active"));
    drainSamples.add(1);
  }
}

const classNames = ["sideways", "growth", "decline"];

function validDigest(value) {
  return typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
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
    stop = interval.stop;
  }
  if (stop !== steps) fail("Direction intervals do not cover the complete grid");
}

function checkResult(result, id, snapshot, resolved) {
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
  const gridMatches = Object.entries(snapshot.grid).every(([key, value]) => result.time_grid[key] === value);
  if (!gridMatches || !Array.isArray(result.instruments) || !Array.isArray(result.sequences) ||
    result.instruments.length !== snapshot.instruments.length ||
    result.sequences.length !== snapshot.instruments.length) {
    fail("Result differs from the input instrument or time grid");
  }
  snapshot.instruments.forEach((instrument, index) => {
    if (result.instruments[index].instrument_id !== instrument) fail("Result instrument differs from input");
    checkSequence(result.sequences[index], instrument, snapshot.grid.step_count);
  });
}

function rejectionCode(response) {
  try {
    const code = response.json("error.code");
    if (typeof code === "string" && /^[a-z0-9_]{1,64}$/.test(code)) return code;
  } catch {
    // A transport failure or a non-JSON error page can have no API error code.
  }
  return "unknown";
}

function submitAndVerify(snapshot) {
  const body = JSON.stringify({ snapshot_id: snapshot.id, model_version_id: model });
  const start = Date.now();
  const response = send("submit analysis", "/analyses", body, sustained ? [202, 429] : [202], crypto.randomUUID());
  if (response.status !== 202) {
    const code = rejectionCode(response);
    if (sustained && response.status === 429 && code === "capacity_limit") {
      quotaRefused.add(1);
      return;
    }
    submitFailed.add(1);
    fail(`Submit analysis returned HTTP ${response.status} (${code})`);
  }
  accepted.add(1);
  intakeTime.add(Date.now() - start);
  let analysis = response.json();
  const id = analysis.analysis_id;
  if (!id) fail("Accepted analysis has no ID");
  console.log(`REGIMEX_ACCEPTED ${JSON.stringify({ phase, analysis_id: id })}`);
  const resolved = analysis.job.model;
  if (analysis.snapshot_id !== snapshot.id || resolved.model_version_id !== model ||
    !validDigest(resolved.bundle_sha256) || !validDigest(resolved.inference_policy_sha256) ||
    (expectedBundle !== undefined && resolved.bundle_sha256 !== expectedBundle) ||
    (expectedPolicy !== undefined && resolved.inference_policy_sha256 !== expectedPolicy)) {
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
    if (Date.now() - start >= resultTimeoutMs) fail(`Analysis ${id} timed out`);
    sleep(0.5);
    analysis = request("poll analysis", path, 200).json();
    if (analysis.analysis_id !== id) fail("Status belongs to a different analysis");
  }
  if (!analysis.result_available) fail("Successful analysis has no result");
  const result = request("download result", path + "/result", 200);
  const elapsed = Date.now() - start;
  if (elapsed >= resultTimeoutMs) fail(`Analysis ${id} timed out`);
  checkResult(result.json(), id, snapshot, resolved);
  resultTime.add(elapsed);
  verified.add(1);
}

export default function (snapshots) {
  started.add(1);
  try {
    submitAndVerify(snapshots[exec.scenario.iterationInTest % snapshots.length]);
    // An arrival is accounted for only after verification or an explicit quota refusal.
    accounted.add(1);
  } catch (error) {
    failed.add(1);
    throw error;
  }
}
