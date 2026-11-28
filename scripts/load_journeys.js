import { fail, sleep } from "k6";
import exec from "k6/execution";
import http from "k6/http";
import { Counter, Trend } from "k6/metrics";
import { checkLaunchSource, checkResult, checkStatus, integer, loadTarget, rejectionCode, validDigest } from "./load_checks.js";

const api = loadTarget();
const model = __ENV.REGIMEX_LOAD_MODEL;
const scenario = __ENV.REGIMEX_LOAD_JOURNEY;
const owners = ["first", "second"];
const tokens = { first: __ENV.REGIMEX_LOAD_TOKEN, second: __ENV.REGIMEX_LOAD_SECOND_TOKEN };
if (!model || model === "active") throw Error("Choose an immutable published model version");
if (!tokens.first || !tokens.second || tokens.first === tokens.second) {
  throw Error("Supply credentials for two different test owners");
}
const actions = { contention, burst, replay, browse, cancel, uploads };
if (!Object.hasOwn(actions, scenario)) throw Error("Unknown load journey");
const burstSize = integer("REGIMEX_LOAD_BURST_PER_OWNER", 4, 12, 2);
const iterations = scenario === "browse" ? 12 : 1;
const raw = {};
const fixtures = JSON.parse(open(`${__ENV.REGIMEX_LOAD_DIR}/../inputs.json`));
for (const kind of Object.keys(fixtures)) raw[kind] = open(`${__ENV.REGIMEX_LOAD_DIR}/${kind}.json`);

const completed = new Counter("journeys_verified");
const drained = new Counter("idle_owners_after_journey");
const requested = new Counter("journey_jobs_requested");
const accepted = new Counter("journey_jobs_accepted");
const verified = new Counter("journey_results_verified");
const refused = new Counter("journey_capacity_refusals");
const cancelled = new Counter("journey_jobs_cancelled");
const uploadLatency = new Trend("journey_upload_ms", true);
const resultLatency = new Trend("journey_submit_to_result_ms", true);
const endToEnd = new Trend("journey_upload_to_result_ms", true);
const outstanding = new Trend("long_jobs_active_at_short_intake");

const thresholds = {
  http_req_failed: ["rate==0"],
  journeys_verified: [`count==${iterations}`],
  idle_owners_after_journey: ["count==2"],
  journey_results_verified: ["count>0"],
};
if (scenario === "cancel") thresholds.journey_jobs_cancelled = ["count>0"];
export const options = {
  scenarios: {
    journey: {
      executor: "shared-iterations", vus: scenario === "browse" ? 2 : 1,
      iterations, maxDuration: "12m", gracefulStop: "10s",
    },
  },
  setupTimeout: "4m",
  teardownTimeout: "20s",
  thresholds,
  maxRedirects: 0,
  batch: 2 * burstSize,
  batchPerHost: 2 * burstSize,
  hosts: { localhost: "127.0.0.1" },
  systemTags: ["name", "method", "status", "scenario", "error_code", "expected_response"],
};

function message(owner, name, path, body = null, statuses = [200], key) {
  const headers = { Authorization: `Bearer ${tokens[owner]}`, "Content-Type": "application/json" };
  if (key) headers["Idempotency-Key"] = key;
  return {
    method: body === null ? "GET" : "POST",
    url: api + path,
    body,
    params: {
      headers, tags: { name, owner }, timeout: "30s",
      responseCallback: http.expectedStatuses(...statuses),
    },
  };
}

function send(request) {
  return http.request(request.method, request.url, request.body, request.params);
}

function request(owner, name, path, body = null, status = 200, key) {
  const response = send(message(owner, name, path, body, [status], key));
  if (response.status !== status) fail(`${name} for ${owner}: HTTP ${response.status}, expected ${status}`);
  return response;
}

function list(owner, name, path) {
  const items = request(owner, name, path).json("items");
  if (!Array.isArray(items)) fail(`${name} did not return items`);
  return items;
}

function savedSnapshot(response, owner, kind, start) {
  if (response.status !== 201) fail(`Upload for ${owner} returned HTTP ${response.status}`);
  const saved = response.json();
  const parsed = JSON.parse(raw[kind]);
  uploadLatency.add(response.timings.duration, { owner, kind });
  if (!saved.snapshot_id || saved.content_sha256 !== parsed.content_sha256) fail("Snapshot differs from uploaded input");
  return {
    id: saved.snapshot_id, digest: saved.content_sha256, grid: parsed.time_grid,
    source: parsed.source, instruments: parsed.instruments, owner, kind, start,
  };
}

function upload(owner, kind) {
  const start = Date.now();
  const response = request(owner, "upload snapshot", "/snapshots", raw[kind], 201);
  return savedSnapshot(response, owner, kind, start);
}

function submission(snapshot, key, allowRefusal = false) {
  const body = JSON.stringify({ snapshot_id: snapshot.id, model_version_id: model });
  return message(snapshot.owner, "submit analysis", "/analyses", body, allowRefusal ? [202, 429, 503] : [202], key);
}

function jobFrom(response, snapshot, start) {
  if (response.status !== 202) fail(`Analysis for ${snapshot.owner} returned HTTP ${response.status}`);
  const view = response.json();
  const pinned = view.job?.model;
  checkLaunchSource(view.job);
  if (!view.analysis_id || !view.job.operation_id || view.snapshot_id !== snapshot.id || view.snapshot_sha256 !== snapshot.digest ||
      pinned?.model_version_id !== model || !validDigest(pinned.bundle_sha256) || !validDigest(pinned.inference_policy_sha256) ||
      (__ENV.REGIMEX_LOAD_EXPECTED_BUNDLE_SHA256 && pinned.bundle_sha256 !== __ENV.REGIMEX_LOAD_EXPECTED_BUNDLE_SHA256) ||
      (__ENV.REGIMEX_LOAD_EXPECTED_POLICY_SHA256 && pinned.inference_policy_sha256 !== __ENV.REGIMEX_LOAD_EXPECTED_POLICY_SHA256)) {
    fail("Accepted analysis differs from requested snapshot or model");
  }
  return { id: view.analysis_id, operation: view.job.operation_id, snapshot, pinned, start };
}

function submit(jobs, snapshot, key = crypto.randomUUID()) {
  const start = Date.now();
  requested.add(1, { owner: snapshot.owner, kind: snapshot.kind });
  const response = send(submission(snapshot, key));
  if (response.status === 202) accepted.add(1, { owner: snapshot.owner, kind: snapshot.kind });
  const job = jobFrom(response, snapshot, start);
  jobs.push(job);
  return job;
}

function checkDeadline(start, now = Date.now()) {
  if (now - start > 180000) fail("Analysis exceeded its three-minute journey deadline");
}

function isDone(job) {
  const { snapshot, id, pinned, start } = job;
  const tags = { owner: snapshot.owner, kind: snapshot.kind };
  const path = `/analyses/${encodeURIComponent(id)}`;
  checkDeadline(start);
  const view = request(snapshot.owner, "poll analysis", path).json();
  checkDeadline(start);
  checkStatus(view, job);
  if (view.job.state === "succeeded") {
    if (!view.result_available) fail("Successful job has no result");
    const response = request(snapshot.owner, "download result", path + "/result");
    const ended = Date.now();
    checkDeadline(start, ended);
    checkResult(response.json(), id, snapshot, pinned, model);
    resultLatency.add(ended - start, tags);
    if (job.includesUpload) endToEnd.add(ended - snapshot.start, tags);
    verified.add(1, tags);
    console.log(`REGIMEX_VERIFIED ${JSON.stringify({ phase: `${scenario}-${snapshot.owner}-${snapshot.kind}`, analysis_id: id })}`);
    return true;
  }
  if (view.job.state === "cancelled" && job.cancelRequested) {
    if (view.result_available || view.report_available) fail("Cancelled job published output");
    cancelled.add(1, tags);
    return true;
  }
  if (view.job.state === "queued" || view.job.state === "running") return false;
  if (job.cancelRequested && view.job.state === "cancel_requested") return false;
  fail(`Unexpected analysis state ${view.job.state}`);
}

function finish(jobs, errors = []) {
  const unique = new Map(jobs.map(job => [job.id, job]));
  if (unique.size !== jobs.length) errors.push("Distinct requests returned a duplicate analysis ID");
  let pending = [...unique.values()];
  while (pending.length) {
    pending = pending.filter(job => {
      try {
        return !isDone(job);
      } catch (error) {
        errors.push(`Analysis ${job.id}: ${error.message}`);
        return false;
      }
    });
    if (pending.length) sleep(0.25);
  }
  if (errors.length) fail(errors.join("\n"));
}

// A later upload or submission can fail after earlier jobs were admitted.
function runAndDrain(start) {
  const jobs = [];
  const errors = [];
  try {
    start(jobs);
  } catch (error) {
    errors.push(error.message);
  }
  finish(jobs, errors);
  return jobs;
}

function active(owner) {
  return list(owner, "active analyses", "/analyses?state=active&limit=100");
}

export function setup() {
  for (const counter of [requested, accepted, verified, cancelled, refused, completed, drained]) counter.add(0);
  const state = {};
  for (const owner of owners) {
    if (active(owner).length) fail("Use idle test owners with no active jobs");
    state[owner] = { anchor: upload(owner, "anchor") };
  }
  if (state.first.anchor.id === state.second.anchor.id) fail("The tokens resolve to the same owner; use distinct accounts");
  if (scenario === "browse") {
    const jobs = runAndDrain(pending => {
      for (const owner of owners) submit(pending, state[owner].anchor);
    });
    for (const job of jobs) state[job.snapshot.owner].job = job;
  }
  return state;
}

function contention() {
  for (const [longOwner, shortOwner] of [["first", "second"], ["second", "first"]]) {
    runAndDrain(jobs => {
      const long = upload(longOwner, "long");
      for (let index = 0; index < 4; index++) submit(jobs, long).includesUpload = index === 0;
      const short = upload(shortOwner, "short");
      submit(jobs, short).includesUpload = true;
      const activeIds = new Set(active(longOwner).map(item => item.analysis_id));
      outstanding.add(jobs.filter(j => j.snapshot === long && activeIds.has(j.id)).length, { owner: shortOwner });
    });
  }
}

function burst(state) {
  const requests = [];
  const snapshots = [];
  for (let i = 0; i < burstSize; i++) {
    for (const owner of owners) {
      snapshots.push(state[owner].anchor);
      requests.push(submission(state[owner].anchor, crypto.randomUUID(), true));
    }
  }
  const start = Date.now();
  const jobs = [];
  const errors = [];
  for (const owner of owners) requested.add(burstSize, { owner, kind: "anchor" });
  for (const [i, response] of http.batch(requests).entries()) {
    const snapshot = snapshots[i];
    if ((response.status === 429 || response.status === 503) && rejectionCode(response) === "capacity_limit") {
      refused.add(1, { owner: snapshot.owner, status: String(response.status) });
      continue;
    }
    if (response.status === 202) accepted.add(1, { owner: snapshot.owner, kind: snapshot.kind });
    try {
      jobs.push(jobFrom(response, snapshot, start));
    } catch (error) {
      errors.push(error.message);
    }
  }
  if (!jobs.length && !errors.length) fail("The entire burst was refused");
  finish(jobs, errors);
  // Fresh keys after the drain measure recovery for both accounts.
  runAndDrain(pending => {
    for (const owner of owners) submit(pending, state[owner].anchor);
  });
}

function replay(state) {
  const anchor = state.first.anchor;
  if (upload("first", "anchor").id !== anchor.id) fail("Repeated snapshot upload created a new snapshot");
  const key = crypto.randomUUID();
  const start = Date.now();
  requested.add(1, { owner: anchor.owner, kind: anchor.kind });
  const duplicate = http.batch([submission(anchor, key), submission(anchor, key)]);
  const [first] = runAndDrain(jobs => {
    const errors = [];
    for (const response of duplicate) {
      try {
        const job = jobFrom(response, anchor, start);
        const original = jobs[0];
        if (!original || original.id !== job.id) {
          jobs.push(job);
          accepted.add(1, { owner: "first", kind: anchor.kind });
        }
        if (original && (original.id !== job.id || original.operation !== job.operation)) {
          errors.push("Concurrent retries created distinct operations");
        }
      } catch (error) {
        errors.push(error.message);
      }
    }
    if (errors.length) fail(errors.join("\n"));
    const alternate = upload("first", "short");
    const conflict = request("first", "changed request replay", "/analyses",
      JSON.stringify({ snapshot_id: alternate.id, model_version_id: model }), 409, key);
    if (rejectionCode(conflict) !== "conflict") fail("Changed request did not report an idempotency conflict");
    submit(jobs, state.second.anchor, key);
  });
  const late = jobFrom(send(submission(anchor, key)), anchor, start);
  if (late.id !== first.id || late.operation !== first.operation) {
    if (late.id !== first.id) accepted.add(1, { owner: anchor.owner, kind: anchor.kind });
    finish(late.id === first.id ? [] : [late], ["Completed request replay changed identity"]);
  }
  const rows = list("first", "snapshot history", `/analyses?snapshot_id=${anchor.id}&limit=100`);
  if (rows.length !== 1 || rows[0].analysis_id !== first.id) fail("Snapshot history contains an extra submission");
}

function browse(state) {
  const owner = owners[exec.scenario.iterationInTest % owners.length];
  const job = state[owner].job;
  const experiments = list(owner, "browse experiments", "/experiments?limit=20");
  const history = list(owner, "browse analyses", `/analyses?snapshot_id=${job.snapshot.id}&limit=20`);
  if (!experiments.some(item => item.snapshot_id === job.snapshot.id) || !history.some(item => item.analysis_id === job.id)) {
    fail("The prepared experiment or analysis is missing from history");
  }
  const detail = request(owner, "analysis details", `/analyses/${job.id}`).json();
  checkStatus(detail, job);
  if (!detail.report_available || !detail.result_available) {
    fail("Workspace detail is incomplete");
  }
  const report = request(owner, "download report", `/analyses/${job.id}/report`);
  if (!String(report.headers["Content-Type"]).includes("text/html") || !/<!doctype html>/i.test(report.body)) {
    fail("Expected an HTML report");
  }
}

function cancel(state) {
  runAndDrain(jobs => {
    const job = submit(jobs, upload("first", "long"));
    const outcome = request("first", "cancel analysis", `/jobs/${job.operation}/cancel`, "{}").json();
    if (outcome.operation_id !== job.operation) fail("Cancelled a different operation");
    job.cancelRequested = true;
    submit(jobs, state.second.anchor);
  });
}

function uploads() {
  const items = owners.flatMap(owner => ["short", "long"].map(kind => ({ owner, kind })));
  const requests = items.map(({ owner, kind }) => message(owner, "concurrent upload", "/snapshots", raw[kind], [201]));
  const start = Date.now();
  const snapshots = http.batch(requests).map((r, i) => savedSnapshot(r, items[i].owner, items[i].kind, start));
  if (new Set(snapshots.map(s => s.id)).size !== items.length) fail("Distinct owner/snapshot pairs shared an ID");
  for (const [i, response] of http.batch(requests).entries()) {
    if (response.status !== 201 || response.json().snapshot_id !== snapshots[i].id) fail("Repeated upload changed snapshot identity");
  }
  runAndDrain(jobs => {
    for (const snapshot of [snapshots[0], snapshots[3]]) submit(jobs, snapshot);
  });
}

export default function (state) {
  actions[scenario](state);
  completed.add(1);
}

export function teardown() {
  for (const owner of owners) {
    if (active(owner).length) fail(`Owner ${owner} still has active jobs`);
    drained.add(1, { owner });
  }
}
