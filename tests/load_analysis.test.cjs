const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

async function loadClient(env = {}, file = "load_analysis.js") {
  const metrics = new Map();
  const clock = { now: () => 0 };
  const http = { expectedStatuses() {} };
  const execution = { scenario: { startTime: 0, iterationInTest: 0 } };
  class Metric {
    constructor(name) {
      this.values = [];
      metrics.set(name, this.values);
    }
    add(value) { this.values.push(value); }
  }

  const context = vm.createContext({
    Date: { now: () => clock.now(), parse: Date.parse }, console: { log() {} },
    __ENV: { REGIMEX_LOAD_PHASE: "sustained", REGIMEX_LOAD_DURATION_SECONDS: "10",
      REGIMEX_LOAD_MODEL: "fixture-model", ...env },
    open: name => name.endsWith("/inputs.json") ? '{"anchor":{},"short":{},"long":{}}' : JSON.stringify({ content_sha256: "ab".repeat(32), instruments: [], time_grid: { step_count: 256 },
      source: { run_id: name } }), crypto: { randomUUID: () => "fixture-key" },
  });
  const dependencies = {
    k6: { fail(message) { throw Error(message); }, sleep() {} },
    "k6/http": { default: http },
    "k6/execution": { default: execution },
    "k6/metrics": { Counter: Metric, Trend: Metric },
  };
  const modules = new Map();
  function local(name) {
    return new vm.SourceTextModule(fs.readFileSync(path.join(__dirname, "../scripts", name), "utf8"), { context });
  }
  const client = local(file);
  await client.link(name => {
    if (!modules.has(name)) {
      if (name === "./load_checks.js") modules.set(name, local(name));
      else {
        const exports = dependencies[name];
        modules.set(name, new vm.SyntheticModule(Object.keys(exports), function () {
          for (const [key, value] of Object.entries(exports)) this.setExport(key, value);
        }, { context }));
      }
    }
    return modules.get(name);
  });
  await client.evaluate();
  return { client: client.namespace, http, clock, metrics, execution };

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

test("the client requires an explicit immutable model before setup", async () => {
  for (const model of [undefined, "", "active"]) {
    await assert.rejects(loadClient({ REGIMEX_LOAD_MODEL: model }), /REGIMEX_LOAD_MODEL/);
  }
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
  const instrument = { instrument_id: "ETH/USD", base_asset: "ETH", quote_asset: "USD",
    initial_price: 100, units: { price_unit: "quote_asset_per_base_asset", volume_unit: "base_asset",
      price_decimal_places: 2, volume_decimal_places: 4 } };
  const snapshot = { id: "snapshot", digest: "ab".repeat(32), grid: { step_count: 2 },
    source: { source_id: "exchange", source_version: "1", configuration_id: "config", run_id: "run-1",
      snapshot_id: "close", state_revision: "1", run_status: "completed", consistency: "atomic_completed_snapshot" },
    instruments: [instrument], lastCreatedAt: null };
  const interval = { instrument_id: "ETH/USD", start: 0, stop: 2, status: "observed", direction: "sideways" };
  const valid = {
    result_version: "2.0", analysis_id: "analysis", snapshot_id: snapshot.id, snapshot_sha256: snapshot.digest,
    model: { model_version_id: "fixture-model", bundle_sha256: "12".repeat(32), inference_policy_sha256: "34".repeat(32) },
    time_grid: snapshot.grid, source: Object.fromEntries(Object.entries(snapshot.source).reverse()),
    class_names: ["sideways", "growth", "decline"],
    instruments: [{ ...instrument }],
    sequences: [{ instrument_id: "ETH/USD", probabilities: [[1, 0, 0], [0.5, 0.2, 0.3]], intervals: [interval] }],
  };
  const apiSource = { channel: "api", command: null };
  function verify(result, intakeSource = apiSource, pollSource = apiSource, changeView = () => {}) {
    http.request = (method, url) => {
      if (url.endsWith("/result")) return { status: 200, json: () => result };
      const intake = method === "POST";
      const job = { operation_id: "job", model: { ...result.model }, created_at: "2026-01-01T00:00:00Z",
        state: intake ? "queued" : "succeeded", launch_source: intake ? intakeSource : pollSource };
      const view = { analysis_id: "analysis", snapshot_id: snapshot.id, snapshot_sha256: snapshot.digest,
        result_available: !intake, job };
      changeView(view, intake);
      return { status: intake ? 202 : 200, json: () => view };
    };
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
    result => { delete result.source; },
    result => { result.source.run_id = "another-run"; },
    result => { result.source.state_revision = "2"; },
    result => { result.instruments[0].quote_asset = "EUR"; },
    result => { result.instruments[0].units.price_decimal_places = 8; },
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
  const invalidSources = [null, { channel: "web", command: null }, { channel: "api", command: "regimex" }];
  for (const source of invalidSources) {
    assert.throws(() => verify(valid, source, apiSource), /launch source/);
    assert.throws(() => verify(valid, apiSource, source), /launch source/);
  }
  const changes = [
    (view, intake) => { if (intake) view.snapshot_sha256 = "ff".repeat(32); },
    (view, intake) => { if (intake) delete view.job.operation_id; },
    (view, intake) => { if (!intake) view.job.operation_id = "another-job"; },
    (view, intake) => { if (!intake) view.snapshot_id = "another-snapshot"; },
    (view, intake) => { if (!intake) view.snapshot_sha256 = "ff".repeat(32); },
    (view, intake) => { if (!intake) view.job.model.model_version_id = "another-model"; },
    (view, intake) => { if (!intake) view.job.model.bundle_sha256 = "ff".repeat(32); },
    (view, intake) => { if (!intake) view.job.model.inference_policy_sha256 = "ff".repeat(32); },
  ];
  for (const change of changes) {
    assert.throws(() => verify(valid, apiSource, apiSource, change), /[Ss]ubmission|[Ss]tatus/);
  }
  assert.deepEqual(metrics.get("analyses_verified"), classifications.map(() => 1));
  assert.equal(metrics.get("analyses_failed").length,
    invalid.length + invalidClassifications.length + 2 * invalidSources.length + changes.length);
});

test("only HTTPS may carry a token to a named load balancer", async () => {
  const {client, http} = await loadClient({ REGIMEX_LOAD_SCHEME: "https",
    REGIMEX_LOAD_HOST: "test.example.org", REGIMEX_LOAD_PORT: "443" });
  http.request = (_method, url) => {
    assert.equal(new URL(url).origin, "https://test.example.org");
    return { status: 200, json: () => [] };
  };
  client.sampleQueue();
  await assert.rejects(loadClient({ REGIMEX_LOAD_HOST: "test.example.org" }), /requires HTTPS/);
  for (const host of ["wrong:9443", "user@test.example.org", "a.example/x", "a.\nb"]) {
    await assert.rejects(loadClient({ REGIMEX_LOAD_SCHEME: "https", REGIMEX_LOAD_HOST: host }), /hostname/);
  }
});

function testJourney(name, values = {}) {
  return loadClient({ REGIMEX_LOAD_JOURNEY: name, REGIMEX_LOAD_TOKEN: "test-first",
    REGIMEX_LOAD_SECOND_TOKEN: "test-second", ...values }, "load_journeys.js");
}

test("burst records refusals and checks admitted jobs after submission and polling failures", async () => {
  const {client, http, metrics} = await testJourney("burst", {REGIMEX_LOAD_BURST_PER_OWNER: "2"});
  const model = {model_version_id: "fixture-model", bundle_sha256: "ab".repeat(32),
    inference_policy_sha256: "cd".repeat(32)};
  const state = Object.fromEntries(["first", "second"].map(owner => [owner, {
    anchor: {owner, kind: "anchor", id: owner, digest: "ef".repeat(32),
      source: {run_id: "burst"}, grid: {step_count: 1}, instruments: []},
  }]));
  const admitted = new Map();
  http.batch = requests => requests.map((request, index) => {
    if (index === 0) return {status: 500};
    if (index === 2) return {status: 429, json: () => "capacity_limit"};
    const snapshot = state[request.params.tags.owner].anchor;
    const id = "job-" + index;
    const view = {analysis_id: id, snapshot_id: snapshot.id, snapshot_sha256: snapshot.digest,
      job: {operation_id: id, state: "queued", model, launch_source: {channel: "api", command: null}}};
    admitted.set(id, view);
    return {status: 202, json: () => view};
  });
  const polled = [];
  http.request = (_method, url, _body, {tags: {name, owner}}) => {
    const id = new URL(url).pathname.split("/")[4];
    const view = admitted.get(id);
    if (name === "poll analysis") {
      polled.push(id);
      return {status: 200, json: () => ({...view, result_available: true,
        job: {...view.job, state: id === "job-1" ? "failed" : "succeeded"}})};
    }
    assert.equal(name, "download result");
    const {source, grid, instruments} = state[owner].anchor;
    return {status: 200, json: () => ({...view, result_version: "2.0", model,
      class_names: ["sideways", "growth", "decline"], source, time_grid: grid, instruments, sequences: []})};
  };
  assert.throws(() => client.default(state), /HTTP 500[\s\S]*job-1: Unexpected analysis state failed/);
  assert.deepEqual(polled, ["job-1", "job-3"]);
  assert.deepEqual(metrics.get("journey_jobs_requested"), [2, 2]);
  assert.deepEqual(metrics.get("journey_jobs_accepted"), [1, 1]);
  assert.deepEqual(metrics.get("journey_capacity_refusals"), [1]);
  assert.deepEqual(metrics.get("journey_results_verified"), [1]);
  assert.deepEqual(metrics.get("journeys_verified"), []);
});

test("journey deadlines include the terminal poll and the result download", async () => {
  for (const [pollEnd, downloadEnd, expectedDownloads] of [
    [189900, 197900, 0], [179950, 187950, 1], [180000, 180000, 2],
  ]) {
    const {client, http, clock, metrics} = await testJourney("browse", {REGIMEX_LOAD_DIR: "inputs"});
    let now = 0, downloads = 0;
    clock.now = () => now;
    const digest = "ab".repeat(32);
    const model = {model_version_id: "fixture-model", bundle_sha256: "cd".repeat(32), inference_policy_sha256: "ef".repeat(32)};
    http.request = (_method, _url, _body, {tags: {name, owner}}) => {
      const id = "job-" + owner;
      const snapshot = "snapshot-" + owner;
      const view = {analysis_id: id, snapshot_id: snapshot, snapshot_sha256: digest,
        job: {operation_id: id, model, launch_source: {channel: "api", command: null}}};
      switch (name) {
        case "active analyses":
          return {status: 200, json: () => []};
        case "upload snapshot":
          return {status: 201, timings: {duration: 1}, json: () => ({snapshot_id: snapshot, content_sha256: digest})};
        case "submit analysis":
          return {status: 202, json: () => view};
        case "poll analysis":
          if (owner === "first" && now < 179900) {
            now = Math.min(179900, now + 10000);
            view.job.state = "running";
          } else if (owner === "second" && !downloads) {
            view.job.state = "running";
          } else {
            if (owner === "first") now = pollEnd;
            view.job.state = "succeeded";
            view.result_available = true;
          }
          return {status: 200, json: () => view};
        case "download result":
          downloads++;
          now = downloadEnd;
          return {status: 200, json: () => ({
            ...view, result_version: "2.0", model, class_names: ["sideways", "growth", "decline"],
            source: {run_id: "inputs/anchor.json"}, time_grid: {step_count: 256}, instruments: [], sequences: [],
          })};
        default: assert.fail(name);
      }
    };
    if (downloadEnd > 180000) {
      assert.throws(() => client.setup(), /deadline/);
      assert.deepEqual(metrics.get("journey_results_verified"), [0]);
      assert.deepEqual(metrics.get("journey_submit_to_result_ms"), []);
    } else {
      client.setup();
      assert.deepEqual(metrics.get("journey_results_verified"), [0, 1, 1]);
      assert.deepEqual(metrics.get("journey_submit_to_result_ms"), [180000, 180000]);
    }
    assert.equal(downloads, expectedDownloads);
  }
});

test("browse setup verifies the first owner's job after the second submission fails", async () => {
  const {client, http, metrics} = await testJourney("browse", {REGIMEX_LOAD_DIR: "inputs"});
  const digest = "ab".repeat(32);
  const model = {model_version_id: "fixture-model", bundle_sha256: digest, inference_policy_sha256: digest};
  const view = {analysis_id: "analysis-1", snapshot_id: "first", snapshot_sha256: digest,
    job: {operation_id: "op-1", model, state: "queued", launch_source: {channel: "api", command: null}}};
  http.request = (_method, _url, _body, {tags: {name, owner}}) => {
    switch (name) {
      case "active analyses": return {status: 200, json: () => []};
      case "upload snapshot": return {status: 201, timings: {duration: 1},
        json: () => ({snapshot_id: owner, content_sha256: digest})};
      case "submit analysis": return owner === "first" ? {status: 202, json: () => view} : {status: 500};
      case "poll analysis": return {status: 200, json: () => ({...view, result_available: true,
        job: {...view.job, state: "succeeded"}})};
      case "download result": return {status: 200, json: () => ({...view, result_version: "2.0", model,
        class_names: ["sideways", "growth", "decline"], source: {run_id: "inputs/anchor.json"},
        time_grid: {step_count: 256}, instruments: [], sequences: []})};
      default: assert.fail(name);
    }
  };
  assert.throws(() => client.setup(), /second returned HTTP 500/);
  assert.deepEqual(metrics.get("journey_jobs_accepted"), [0, 1]);
  assert.deepEqual(metrics.get("journey_results_verified"), [0, 1]);
});

test("two-owner journeys require distinct owners even when the tokens differ", async () => {
  await assert.rejects(testJourney("contention", {REGIMEX_LOAD_SECOND_TOKEN: "test-first"}), /distinct|different/);
  const { client, http } = await testJourney("contention");
  let uploads = 0;
  http.request = (method, url, _body, params) => {
    if (method === "GET") return { status: 200, json: field => field === "items" ? [] : null };
    assert.ok(url.endsWith("/snapshots"));
    assert.equal(params.headers.Authorization, uploads++ === 0 ? "Bearer test-first" : "Bearer test-second");
    return { status: 201, timings: { duration: 1 }, json: () => ({ snapshot_id: "same-snapshot", content_sha256: "ab".repeat(32) }) };
  };
  assert.throws(() => client.setup(), /same owner/);
  assert.equal(uploads, 2);
});

test("a retry must return the original analysis and operation even with two HTTP 202s", async () => {
  const { client, http } = await testJourney("replay");
  const snap = { owner: "first", id: "first-anchor", kind: "anchor", digest: "ab".repeat(32) };
  http.request = () => ({ status: 201, timings: { duration: 1 }, json: () => ({ snapshot_id: snap.id, content_sha256: snap.digest }) });
  http.batch = requests => {
    assert.equal(requests.length, 2);
    assert.equal(requests[0].params.headers["Idempotency-Key"], requests[1].params.headers["Idempotency-Key"]);
    return requests.map((_, index) => ({ status: 202, json: () => ({
      analysis_id: "same", snapshot_id: snap.id, snapshot_sha256: snap.digest,
      job: { operation_id: String(index), launch_source: { channel: "api", command: null },
        model: {model_version_id:"fixture-model",bundle_sha256:"cd".repeat(32),inference_policy_sha256:"ef".repeat(32)} },
    }) }));
  };
  assert.throws(() => client.default({ first: { anchor: snap }, second: {anchor:{...snap, owner:"second"}} }), /distinct operations/);
});

test("a journey thresholds all iterations and checks both owners after drain", async () => {
  const {client, http, metrics} = await testJourney("browse");
  assert.deepEqual(Array.from(client.options.thresholds.journeys_verified), ["count==12"]);
  assert.deepEqual(Array.from(client.options.thresholds.idle_owners_after_journey), ["count==2"]);
  http.request = (_method, _url, _body, params) => ({ status:200, json: () => params.tags.owner === "second" ? ["pending"] : [] });
  assert.throws(() => client.teardown(), /second still has active jobs/);
  assert.deepEqual(metrics.get("idle_owners_after_journey"), [1]);
});

test("concurrent uploads verify distinct owner/fixture IDs and stable IDs on replay", async () => {
  const {client, http} = await testJourney("uploads");
  let batches = 0;
  http.batch = requests => {
    assert.equal(requests.length, 4);
    assert.deepEqual(Array.from(requests, r => r.params.tags.owner), ["first","first","second","second"]);
    batches++;
    return requests.map((_, i) => ({status:201, timings: {duration:4}, json: () => ({
      snapshot_id: batches === 2 && i === 3 ? "incorrect-replay" : String(i), content_sha256:"ab".repeat(32),
    })}));
  };
  assert.throws(() => client.default({}), /Repeated upload changed snapshot identity/);
  assert.equal(batches, 2);
});
