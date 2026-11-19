const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

async function loadClient(env = {}) {
  const metrics = new Map();
  const clock = { now: () => 0 };
  const http = { expectedStatuses() {} };
  class Metric {
    constructor(name) {
      this.values = [];
      metrics.set(name, this.values);
    }
    add(value) { this.values.push(value); }
  }

  const context = vm.createContext({
    Date: { now: () => clock.now(), parse: Date.parse }, console: { log() {} },
    __ENV: { REGIMEX_LOAD_PHASE: "sustained", REGIMEX_LOAD_DURATION_SECONDS: "10", ...env },
    open: () => "{}", crypto: { randomUUID: () => "fixture-key" },
  });
  const dependencies = {
    k6: { fail(message) { throw Error(message); }, sleep() {} },
    "k6/http": { default: http },
    "k6/execution": { default: { scenario: { startTime: 0, iterationInTest: 0 } } },
    "k6/metrics": { Counter: Metric, Trend: Metric },
  };
  const client = new vm.SourceTextModule(fs.readFileSync(
    path.join(__dirname, "../scripts/load_analysis.js"), "utf8"), { context });
  await client.link(name => {
    const exports = dependencies[name];
    return new vm.SyntheticModule(Object.keys(exports), function () {
      for (const [key, value] of Object.entries(exports)) this.setExport(key, value);
    }, { context });
  });
  await client.evaluate();
  return { client: client.namespace, http, clock, metrics };
}

test("load requests stay on loopback for the HTTP stand and HTTPS Compose service", async () => {
  for (const [env, origin] of [
    [{}, "http://127.0.0.1:8001"],
    [{ REGIMEX_LOAD_SCHEME: "https" }, "https://localhost:8443"],
    [{ REGIMEX_LOAD_SCHEME: "https", REGIMEX_LOAD_PORT: "8444" }, "https://localhost:8444"],
  ]) {
    const { client, http } = await loadClient(env);
    const urls = [];
    http.request = (_method, url) => {
      urls.push(url);
      return { status: 200, json: () => [] };
    };
    client.sampleQueue();
    assert.ok(urls.length > 0);
    for (const url of urls) assert.equal(new URL(url).origin, origin);
    assert.equal(client.options.hosts.localhost, "127.0.0.1");
    assert.equal(client.options.maxRedirects, 0);
  }
  await assert.rejects(loadClient({ REGIMEX_LOAD_SCHEME: "ftp" }), /must be http or https/);
});

test("sustained queue thresholds require completed polls in both intake halves and after drain", async () => {
  const { client, http, clock, metrics } = await loadClient();
  let time = 0;
  let responseTime = 0;
  clock.now = () => time;
  http.request = () => {
    time = responseTime;
    return { status: 200, json: () => ["queued"] };
  };
  for (responseTime of [4999, 5000, 10000]) client.sampleQueue();
  assert.deepEqual(metrics.get("analysis_queue_first_half_samples"), [1]);
  assert.deepEqual(metrics.get("analysis_queue_second_half_samples"), [1]);
  assert.deepEqual(metrics.get("analysis_queued_first_half"), [1]);
  assert.deepEqual(metrics.get("analysis_queued_second_half"), [1]);
  for (const half of ["first", "second"]) {
    assert.equal(client.options.thresholds[`analysis_queue_${half}_half_samples`][0], "count>0");
  }
  http.request = () => ({ status: 200, json: () => null });
  assert.throws(() => client.teardown(), /Missing analysis list/);
  assert.deepEqual(metrics.get("analysis_after_drain_samples"), []);
  http.request = () => ({ status: 200, json: () => [] });
  client.teardown();
  assert.deepEqual(metrics.get("analysis_active_after_drain"), [0]);
  assert.deepEqual(metrics.get("analysis_after_drain_samples"), [1]);
  assert.equal(client.options.thresholds.analysis_after_drain_samples[0], "count==1");
});

test("a verified result needs the resolved model and a complete classification on every input bin", async () => {
  const { client, http, metrics } = await loadClient({
    REGIMEX_LOAD_EXPECTED_BUNDLE_SHA256: "12".repeat(32),
    REGIMEX_LOAD_EXPECTED_POLICY_SHA256: "34".repeat(32),
  });
  const snapshot = { id: "snapshot", digest: "ab".repeat(32), grid: { step_count: 2 },
    instruments: ["ETH/USD"], lastCreatedAt: null };
  const interval = { instrument_id: "ETH/USD", start: 0, stop: 2, status: "observed", direction: "sideways" };
  const valid = {
    result_version: "2.0", analysis_id: "analysis", snapshot_id: snapshot.id, snapshot_sha256: snapshot.digest,
    model: { model_version_id: "tcn-0.2.0", bundle_sha256: "12".repeat(32), inference_policy_sha256: "34".repeat(32) },
    time_grid: snapshot.grid, class_names: ["sideways", "growth", "decline"], instruments: [{ instrument_id: "ETH/USD" }],
    sequences: [{ instrument_id: "ETH/USD", probabilities: [[1, 0, 0], [0.5, 0.2, 0.3]], intervals: [interval] }],
  };
  function verify(result) {
    http.request = (method, url) => url.endsWith("/result")
      ? { status: 200, json: () => result }
      : { status: 202, json: () => ({ analysis_id: "analysis", snapshot_id: snapshot.id, result_available: true,
        job: { model: result.model, created_at: "2026-01-01T00:00:00Z", state: "succeeded" } }) };
    client.default([snapshot]);
  }
  const classifications = [
    ["observed", "sideways"], ["observed", "growth"], ["observed", "decline"],
    ["no_trades", null], ["insufficient_data", null],
  ];
  for (const [status, direction] of classifications) {
    const result = structuredClone(valid);
    Object.assign(result.sequences[0].intervals[0], { status, direction });
    verify(result);
  }
  const invalid = [
    result => { delete result.model.bundle_sha256; },
    result => { result.model.inference_policy_sha256 = "invalid"; },
    result => { result.model.bundle_sha256 = "aa".repeat(32); },
    result => { result.model.inference_policy_sha256 = "aa".repeat(32); },
    result => { result.result_version = "1"; },
    result => { result.class_names.reverse(); },
    result => { result.instruments[0].instrument_id = "other"; },
    result => { result.sequences[0].probabilities = "xx"; },
    result => { result.sequences[0].probabilities[0] = [0.5, 0.5]; },
    result => { result.sequences[0].probabilities[0] = [1, -0.1, 0.1]; },
    result => { result.sequences[0].probabilities[0] = [NaN, 0, 0]; },
    result => { result.sequences[0].probabilities[0] = [0.3, 0.3, 0.3]; },
    result => { result.sequences[0].intervals[0].start = 1; },
    result => { result.sequences[0].intervals[0].stop = 1; },
    result => { result.sequences[0].intervals[0].instrument_id = "other"; },
  ];
  for (const corrupt of invalid) {
    const result = structuredClone(valid);
    corrupt(result);
    assert.throws(() => verify(result));
  }
  const invalidClassifications = [
    {}, { direction: "growth" }, { status: "missing", direction: null },
    { status: "observed", direction: null }, { status: "observed", direction: "unknown" },
    { status: "no_trades", direction: "growth" }, { status: "insufficient_data" },
  ];
  for (const classification of invalidClassifications) {
    const result = structuredClone(valid);
    result.sequences[0].intervals = [{ instrument_id: "ETH/USD", start: 0, stop: 2, ...classification }];
    assert.throws(() => verify(result), /interval/);
  }
  assert.deepEqual(metrics.get("analyses_verified"), classifications.map(() => 1));
  assert.equal(metrics.get("analyses_failed").length, invalid.length + invalidClassifications.length);
});
