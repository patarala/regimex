"use strict";

// Execute the shipped controller with controlled DOM events and HTTP responses.
// Chromium checks separately cover layout, real DOM behavior and offline reports.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const root = path.resolve(__dirname, "..");
const source = fs.readFileSync(path.join(root, "src/regimex/web/app.js"), "utf8");
const markup = fs.readFileSync(path.join(root, "src/regimex/web/index.html"), "utf8");
const session = {principal: {owner_id: "test-owner"}, csrf_token: "test-csrf"};
const id = (number) => `00000000-0000-0000-0000-${String(number).padStart(12, "0")}`;
const flush = async () => { for (let i = 0; i < 4; i++) await new Promise(setImmediate); };

class Element {
  constructor() {
    this.children = [];
    this.listeners = new Map();
    this.hidden = false;
    this.disabled = false;
    this.value = "";
    this.textContent = "";
    this.dataset = {};
    this.style = {};
    this.classList = {toggle() {}, add() {}, remove() {}};
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  removeAttribute(name) { delete this[name]; }
  setAttribute(name, value) { this[name] = value; }
  addEventListener(name, handler) {
    if (!this.listeners.has(name)) this.listeners.set(name, []);
    this.listeners.get(name).push(handler);
  }
  emit(name, event = {}) {
    return Promise.all((this.listeners.get(name) || []).map((handler) => handler(event)));
  }
  click() {
    return this.disabled || this.hidden ? Promise.resolve() : this.emit("click");
  }
}

function response(value, status = 200) {
  return {ok: status >= 200 && status < 300, status, json: async () => value};
}

function workspace(count = 45) {
  const elements = new Map();
  for (const tag of markup.matchAll(/<[^>]+\bid="([^"]+)"[^>]*>/g)) {
    const node = new Element();
    node.hidden = /\shidden(?:\s|>)/.test(tag[0]);
    node.disabled = /\sdisabled(?:\s|>)/.test(tag[0]);
    elements.set(tag[1], node);
  }
  const get = (name) => {
    assert.ok(elements.has(name), `Unknown DOM element: ${name}`);
    return elements.get(name);
  };
  const calls = [];
  const timers = new Map();
  const location = {href: "http://localhost/"};
  const history = {replaceState: (_state, _title, url) => { location.href = String(url); }};
  const state = {holdHistory: false};
  const items = Array.from({length: count}, (_, index) => ({
    analysis_id: id(index + 1), job: {state: "running", created_at: "2026-01-01T00:00:00Z"},
  }));
  let timerId = 0;
  const context = {
    document: {getElementById: get, createElement: () => new Element()},
    window: {history, addEventListener() {}}, location, URL, console,
    crypto: {randomUUID: () => "request-key"},
    setTimeout: (callback) => { timers.set(++timerId, callback); return timerId; },
    clearTimeout: (number) => timers.delete(number),
    fetch: (url, options) => {
      let resolve;
      const promise = new Promise((done) => { resolve = done; });
      const call = {url, options, settled: false, reply(value, status = 200) {
        assert.equal(this.settled, false);
        this.settled = true;
        resolve(response(value, status));
      }};
      calls.push(call);
      if (url === "/api/v1/session") call.reply(session);
      else if (url.startsWith("/api/v1/analyses?") && !state.holdHistory) {
        const query = new URL(url, location.href).searchParams;
        const offset = Number(query.get("offset"));
        call.reply({items: items.slice(offset, offset + Number(query.get("limit")))});
      }
      return promise;
    },
  };
  vm.runInNewContext(source, context, {filename: "app.js"});
  return {
    get, calls, state, location, timers,
    select: (index) => get("history-list").children[index].children[0].click(),
    pending: (url) => {
      const call = calls.find((item) => item.url === url && !item.settled);
      assert.ok(call, `No pending request: ${url}`);
      return call;
    },
    rows: () => get("history-list").children.map((row) => row.children[0].dataset.analysisId),
  };
}

function metadata(job, state = "running") {
  return {
    result_available: state === "succeeded", report_available: state === "succeeded",
    job: {operation_id: job, state, stage: "predicting", failure: null, model: {model_version_id: "test-model"}},
  };
}

test("selection clears cancellation before loading and after a load failure", async () => {
  const ui = workspace();
  await flush();
  void ui.select(0);
  ui.pending(`/api/v1/analyses/${id(1)}`).reply(metadata("job-A"));
  await flush();
  assert.equal(ui.get("cancel").hidden, false);
  assert.equal(ui.get("history-list").children[0].children[0]["aria-current"], "true");
  void ui.select(1);
  assert.equal(ui.get("cancel").hidden, true);
  assert.equal(ui.get("cancel").disabled, true);
  assert.equal(ui.get("analysis-identity").textContent, id(2));
  assert.equal(ui.get("history-list").children[0].children[0]["aria-current"], undefined);
  assert.equal(ui.get("history-list").children[1].children[0]["aria-current"], "true");
  // Even a synthetic event on the disabled button cannot target the prior job.
  await ui.get("cancel").emit("click");
  ui.pending(`/api/v1/analyses/${id(2)}`).reply({error: {message: "Load failed"}}, 503);
  await flush();
  await ui.get("cancel").emit("click");
  assert.equal(ui.calls.filter((call) => call.url.endsWith("/cancel")).length, 0);
  assert.equal(ui.get("notice").textContent, "Load failed");
});

test("late cancellation responses cannot change another analysis or its actions", async () => {
  const ui = workspace();
  await flush();
  void ui.select(0);
  ui.pending(`/api/v1/analyses/${id(1)}`).reply(metadata("job-A"));
  await flush();
  void ui.get("cancel").click();
  const cancelling = ui.pending("/api/v1/jobs/job-A/cancel");
  void ui.select(1);
  ui.pending(`/api/v1/analyses/${id(2)}`).reply(metadata("job-B"));
  await flush();
  cancelling.reply({error: {message: "Old cancellation failed"}}, 503);
  await flush();
  assert.equal(ui.get("notice").hidden, true);
  assert.equal(ui.get("cancel").disabled, false);
  assert.equal(ui.get("analysis-identity").textContent.startsWith(id(2)), true);
  void ui.get("cancel").click();
  assert.equal(ui.pending("/api/v1/jobs/job-B/cancel").options.method, "POST");
  assert.equal(ui.calls.filter((call) => call.url === `/api/v1/analyses/${id(2)}`).length, 1);
});

test("returning to the same analysis ignores a response from its earlier selection", async () => {
  const ui = workspace();
  await flush();
  void ui.select(0);
  const stale = ui.pending(`/api/v1/analyses/${id(1)}`);
  void ui.select(1);
  void ui.select(0);
  const latest = ui.calls.filter((call) => call.url === `/api/v1/analyses/${id(1)}`).at(-1);
  latest.reply(metadata("new-job", "queued"));
  await flush();
  stale.reply(metadata("old-job", "failed"));
  ui.pending(`/api/v1/analyses/${id(2)}`).reply(metadata("other-job", "failed"));
  await flush();
  assert.equal(ui.get("job-state").textContent, "queued");
  void ui.get("cancel").click();
  assert.ok(ui.pending("/api/v1/jobs/new-job/cancel"));
});

test("cancellation invalidates an earlier in-flight poll of the same analysis", async () => {
  const ui = workspace();
  await flush();
  void ui.select(0);
  ui.pending(`/api/v1/analyses/${id(1)}`).reply(metadata("job-A"));
  await flush();
  void Array.from(ui.timers.values())[0]();
  const stale = ui.pending(`/api/v1/analyses/${id(1)}`);
  void ui.get("cancel").click();
  await ui.get("cancel").emit("click");
  assert.equal(ui.calls.filter((call) => call.url.endsWith("/cancel")).length, 1);
  ui.pending("/api/v1/jobs/job-A/cancel").reply({});
  await flush();
  const latest = ui.calls.filter((call) => call.url === `/api/v1/analyses/${id(1)}`).at(-1);
  latest.reply(metadata("job-A", "cancel_requested"));
  await flush();
  stale.reply(metadata("job-A", "running"));
  await flush();
  assert.equal(ui.get("cancel").hidden, true);
  assert.equal(ui.get("job-state").textContent, "cancel requested");
});

test("a late saved-result response cannot install the previous report", async () => {
  const ui = workspace();
  await flush();
  void ui.select(0);
  ui.pending(`/api/v1/analyses/${id(1)}`).reply(metadata("job-A", "succeeded"));
  await flush();
  const result = ui.pending(`/api/v1/analyses/${id(1)}/result`);
  void ui.select(1);
  ui.pending(`/api/v1/analyses/${id(2)}`).reply(metadata("job-B"));
  result.reply({unrelated: "must never render"});
  await flush();
  assert.equal(ui.get("result").hidden, true);
  assert.equal(ui.get("download-report").hidden, true);
  assert.equal(ui.get("report-preview").src, undefined);
  assert.equal(ui.get("job-state").textContent, "running");
});

test("history pages reach every analysis without accumulating unbounded DOM rows", async () => {
  const ui = workspace(45);
  await flush();
  assert.deepEqual(ui.rows(), Array.from({length: 20}, (_, index) => id(index + 1)));
  assert.equal(ui.get("history-previous").disabled, true);
  await ui.get("history-next").click();
  assert.deepEqual(ui.rows(), Array.from({length: 20}, (_, index) => id(index + 21)));
  assert.equal(ui.get("history-range").textContent, "21–40");
  await ui.get("history-next").click();
  assert.deepEqual(ui.rows(), [41, 42, 43, 44, 45].map(id));
  assert.equal(ui.get("history-next").disabled, true);
  await ui.get("history-previous").click();
  assert.equal(ui.rows()[0], id(21));
  await ui.get("refresh-history").click();
  assert.equal(ui.rows()[0], id(1));
  assert.equal(ui.get("history-range").textContent, "1–20");
});

test("late history pages cannot replace a newer refresh", async () => {
  const ui = workspace();
  await flush();
  ui.state.holdHistory = true;
  void ui.get("history-next").click();
  const stale = ui.pending("/api/v1/analyses?limit=21&offset=20");
  void ui.get("refresh-history").click();
  ui.pending("/api/v1/analyses?limit=21&offset=0").reply({items: []});
  await flush();
  stale.reply({items: [{analysis_id: id(21), job: {state: "running", created_at: "2026-01-01"}}]});
  await flush();
  assert.deepEqual(ui.rows(), []);
  assert.equal(ui.get("history-empty").hidden, false);
  assert.equal(ui.get("history-next").disabled, true);
  assert.equal(ui.get("history-previous").disabled, true);
});

test("an older invalid-file response cannot replace the selected valid file", async () => {
  const ui = workspace();
  await flush();
  const file = (name) => ({name, size: 24, text: async () => '{"schema_version":"2.1"}'});
  void ui.get("export-file").emit("change", {target: {files: [file("old.json")]}});
  await flush();
  const old = ui.pending("/api/v1/validate");
  void ui.get("export-file").emit("change", {target: {files: [file("new.json")]}});
  await flush();
  const latest = ui.calls.filter((call) => call.url === "/api/v1/validate").at(-1);
  latest.reply({source_id: "test", run_id: "new", step_count: 1, instrument_ids: ["BTC/USD"]});
  await flush();
  old.reply({error: {message: "Old file invalid"}}, 422);
  await flush();
  assert.equal(ui.get("file-name").textContent, "new.json");
  assert.equal(ui.get("classify").disabled, false);
  assert.equal(ui.get("notice").hidden, true);
});

test("an old authentication error cannot reset a newly opened session", async () => {
  const ui = workspace();
  await flush();
  void ui.select(0);
  let finishBody;
  const delayedBody = new Promise((resolve) => { finishBody = resolve; });
  ui.pending(`/api/v1/analyses/${id(1)}`).reply(delayedBody, 401);
  await flush();
  await ui.get("logout").click();
  ui.get("token").value = "new-test-token";
  await ui.get("login-form").emit("submit", {preventDefault() {}});
  finishBody({error: {message: "Old session expired"}});
  await flush();
  assert.equal(ui.get("workspace").hidden, false);
  assert.equal(ui.get("login-panel").hidden, true);
  assert.equal(ui.get("owner").textContent, "test-owner");
  assert.equal(ui.get("notice").hidden, true);
});

test("offline chart rendering removes the PNG export control", async () => {
  const elements = new Map();
  const get = (name) => {
    if (!elements.has(name)) elements.set(name, new Element());
    return elements.get(name);
  };
  const observations = {
    last_price: [100], best_bid: [99], best_ask: [101],
    trade_count: [1], buy_order_count: [1], sell_order_count: [1],
    trade_volume: [1], buy_order_volume: [1], sell_order_volume: [1],
  };
  get("report-data").textContent = JSON.stringify({
    kind: "reference", experiment_hash: "a".repeat(64), analysis_config: {},
    samples: [{experiment_id: "one", instrument_id: "BTC/USD", base_asset: "BTC", quote_asset: "USD",
      steps: [0], intervals: [], observations, source: {source_id: "fixture", source_version: "1",
        snapshot_id: "one", run_id: "one"}, time_grid: {origin: 0, bin_width: 1, completed_horizon: 1, unit: "step"}}],
  });
  get("experiment").value = "one";
  get("sample").value = "0";
  const configurations = [];
  vm.runInNewContext(fs.readFileSync(path.join(root, "src/regimex/templates/direction-report.js"), "utf8"), {
    document: {getElementById: get, createElement: () => new Element(), createTextNode: (text) => text},
    Plotly: {react: async (_node, _traces, _layout, config) => configurations.push(config)},
    history: {replaceState() {}}, location: {hash: ""}, URLSearchParams,
  });
  await flush();
  assert.equal(configurations.length, 1);
  assert.ok(Array.from(configurations[0].modeBarButtonsToRemove).includes("toImage"));
  assert.equal(configurations[0].toImageButtonOptions, undefined);
});
