"use strict";

// Execute the shipped controller with controlled DOM events and HTTP responses.
// Chromium checks separately cover layout, real DOM behavior and offline reports.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const root = path.resolve(__dirname, "..");
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
  return {ok: status >= 200 && status < 300, status, json: async () => structuredClone(await value)};
}

async function loadWorkspace(context) {
  const directory = path.join(root, "src/regimex/web");
  const sandbox = vm.createContext(context);
  const modules = new Map();
  function load(filename) {
    assert.equal(path.dirname(filename), directory, "Workspace imports stay within bundled assets");
    if (!modules.has(filename)) {
      modules.set(filename, new vm.SourceTextModule(fs.readFileSync(filename, "utf8"), {
        context: sandbox, identifier: filename,
      }));
    }
    return modules.get(filename);
  }
  const entry = load(path.join(directory, "app.js"));
  await entry.link((specifier, parent) => load(path.resolve(path.dirname(parent.identifier), specifier)));
  await entry.evaluate();
}

async function workspace(count = 45) {
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
  const state = {
    holdHistory: false, holdModels: false, activeModelId: "tcn-1",
    now: Date.parse("2026-01-01T00:00:30Z"),
  };
  const models = [{model_version_id: "baseline-1"}, {model_version_id: "tcn-1"}];
  const items = Array.from({length: count}, (_, index) => ({
    analysis_id: id(index + 1), snapshot_id: id(1000 + index),
    run_id: `run-${index + 1}`, source_id: "market", result_available: false,
    instrument_ids: [index % 2 ? "ETH/USD" : "BTC/USD"],
    job: {state: "running", created_at: "2026-01-01T00:00:00Z", model: {model_version_id: "tcn-1"}},
  }));
  let timerId = 0;
  let requestId = 0;
  const context = {
    document: {getElementById: get, createElement: () => new Element()},
    window: {history, addEventListener() {}}, location, URL, URLSearchParams, console,
    Date: class extends Date { static now() { return state.now; } },
    crypto: {randomUUID: () => `request-key-${++requestId}`},
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
      else if (url.startsWith("/api/v1/models?") && !state.holdModels) {
        const query = new URL(url, location.href).searchParams;
        assert.equal(query.get("published_only"), "true");
        const offset = Number(query.get("offset"));
        call.reply({items: models.slice(offset, offset + Number(query.get("limit"))), active_model_version_id: state.activeModelId});
      }
      else if (url.startsWith("/api/v1/analyses?") && !state.holdHistory) {
        const query = new URL(url, location.href).searchParams;
        const offset = Number(query.get("offset"));
        const term = (query.get("q") || "").toLowerCase();
        const state = query.get("state");
        const snapshot = query.get("snapshot_id");
        const filtered = items.filter((item) =>
          [item.run_id, item.source_id, ...item.instrument_ids, item.job.model.model_version_id, item.analysis_id].join(" ").toLowerCase().includes(term)
          && (!snapshot || item.snapshot_id === snapshot)
          && (!state || (state === "active" ? ["queued", "running", "cancel_requested"].includes(item.job.state) : item.job.state === state)));
        call.reply({items: filtered.slice(offset, offset + Number(query.get("limit")))});
      }
      return promise;
    },
  };
  await loadWorkspace(context);
  return {
    get, calls, state, location, timers, items, models,
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
    snapshot_id: id(1000),
    result_available: state === "succeeded", report_available: state === "succeeded",
    job: {operation_id: job, state, stage: "predicting", failure: null, model: {model_version_id: "test-model"}},
  };
}

test("selection clears cancellation before loading and after a load failure", async () => {
  const ui = await workspace();
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
  const ui = await workspace();
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
  const ui = await workspace();
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
  const ui = await workspace();
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
  const ui = await workspace();
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
  const ui = await workspace(45);
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
  const ui = await workspace();
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

test("history shows experiment and model names and filters all pages on the server", async () => {
  const ui = await workspace(60);
  await flush();
  const row = ui.get("history-list").children[0].children[0];
  assert.equal(row.children[0].textContent, "BTC/USD");
  assert.equal(row.children[1].textContent, "tcn-1");
  ui.get("history-search").value = "BTC/USD";
  await ui.get("history-filters").emit("submit", {preventDefault() {}});
  assert.equal(ui.rows().length, 20);
  assert.equal(ui.rows()[19], id(39));
  await ui.get("history-next").click();
  assert.deepEqual(ui.rows(), Array.from({length: 10}, (_, index) => id(41 + 2 * index)));
  assert.equal(ui.get("history-range").textContent, "21–30");
  assert.equal(ui.get("history-next").disabled, true);
  ui.get("history-state").value = "failed";
  await ui.get("history-state").emit("change", {preventDefault() {}});
  assert.deepEqual(ui.rows(), []);
  assert.equal(ui.get("history-empty").textContent, "No analyses match these filters.");
  assert.equal(ui.get("history-pages").hidden, true);
  await ui.get("history-clear").click();
  assert.equal(ui.rows()[0], id(1));
  assert.equal(ui.get("history-search").value, "");
  assert.equal(ui.get("history-state").value, "");
  assert.equal(ui.get("history-clear").hidden, true);
});

test("late search responses cannot replace results for a newer query", async () => {
  const ui = await workspace();
  await flush();
  ui.state.holdHistory = true;
  ui.get("history-search").value = "ETH/USD";
  void ui.get("history-filters").emit("submit", {preventDefault() {}});
  const stale = ui.pending("/api/v1/analyses?limit=21&offset=0&q=ETH%2FUSD");
  ui.get("history-search").value = "BTC/USD";
  void ui.get("history-filters").emit("submit", {preventDefault() {}});
  ui.pending("/api/v1/analyses?limit=21&offset=0&q=BTC%2FUSD").reply({items: [ui.items[0]]});
  await flush();
  stale.reply({items: [ui.items[1]]});
  await flush();
  assert.deepEqual(ui.rows(), [id(1)]);
  assert.equal(ui.get("history-loading").hidden, true);
  assert.equal(ui.get("history-list")["aria-busy"], "false");
});

test("a failed selection leaves an active-only history without closing its details", async () => {
  const ui = await workspace(1);
  await flush();
  ui.get("history-state").value = "active";
  await ui.get("history-state").emit("change", {preventDefault() {}});
  void ui.select(0);
  ui.items[0].job.state = "failed";
  const value = metadata("job-A", "failed");
  value.job.failure = {code: "capacity_limit", message: "Time limit exceeded."};
  ui.pending(`/api/v1/analyses/${id(1)}`).reply(value);
  await flush();
  assert.deepEqual(ui.rows(), []);
  assert.equal(ui.get("analysis-panel").hidden, false);
  assert.equal(ui.get("job-detail").textContent, "Time limit exceeded.");
  assert.equal(ui.get("history-empty").textContent, "No analyses match these filters.");
});

test("an in-flight history response cannot undo a terminal state observed by polling", async () => {
  const ui = await workspace(1);
  await flush();
  void ui.select(0);
  ui.pending(`/api/v1/analyses/${id(1)}`).reply(metadata("job-A"));
  await flush();
  ui.state.holdHistory = true;
  void ui.get("refresh-history").click();
  const stale = ui.pending("/api/v1/analyses?limit=21&offset=0");
  void Array.from(ui.timers.values())[0]();
  ui.pending(`/api/v1/analyses/${id(1)}`).reply(metadata("job-A", "failed"));
  await flush();
  const fresh = ui.calls.filter((call) => call.url === "/api/v1/analyses?limit=21&offset=0").at(-1);
  assert.notEqual(fresh, stale);
  fresh.reply({items: [{...ui.items[0], job: {...ui.items[0].job, state: "failed"}}]});
  await flush();
  stale.reply({items: ui.items});
  await flush();
  assert.match(ui.get("history-list").children[0].children[0].children.at(-1).textContent, /^Failed/);
  assert.equal(ui.get("history-list").children[0].children[0]["aria-current"], "true");
});

test("an older invalid-file response cannot replace the selected valid file", async () => {
  const ui = await workspace();
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

test("failed analyses show a reason and next action without offering an exhausted retry", async () => {
  const ui = await workspace();
  await flush();
  void ui.select(0);
  const failure = metadata("job-A", "failed");
  Object.assign(failure.job, {
    attempt: 3, max_attempts: 3,
    failure: {code: "infrastructure_failure", message: "Storage was unavailable.", retryable: true},
  });
  ui.pending(`/api/v1/analyses/${id(1)}`).reply(failure);
  await flush();
  assert.equal(ui.get("job-detail").textContent, "Storage was unavailable.");
  assert.equal(ui.get("job-guidance").hidden, false);
  assert.match(ui.get("job-guidance").textContent, /new analysis/);
  assert.match(ui.get("analysis-identity").textContent, /infrastructure_failure.*3\/3/);
  assert.equal(ui.get("retry").hidden, true);
  assert.equal(ui.get("cancel").hidden, true);
  assert.equal(ui.timers.size, 0);
  void ui.select(1);
  assert.equal(ui.get("job-guidance").hidden, true);
  assert.equal(ui.get("job-guidance").textContent, "");
});

test("validation errors identify the rejected field with bounded readable details", async () => {
  const ui = await workspace();
  await flush();
  const file = {name: "bad.json", size: 24, text: async () => '{"schema_version":"2.1"}'};
  void ui.get("export-file").emit("change", {target: {files: [file]}});
  await flush();
  ui.pending("/api/v1/validate").reply({error: {
    code: "invalid_data", message: "Request validation failed.",
    details: Array.from({length: 7}, (_, index) => ({
      location: ["body", "observations", index, "last_price"], message: "Input should be greater than 0",
    })),
  }}, 422);
  await flush();
  assert.match(ui.get("notice").textContent, /observations\.0\.last_price: Input should be greater than 0/);
  assert.match(ui.get("notice").textContent, /2 more validation errors/);
  assert.doesNotMatch(ui.get("notice").textContent, /observations\.5/);
  assert.equal(ui.get("classify").disabled, true);
});

test("an old authentication error cannot reset a newly opened session", async () => {
  const ui = await workspace();
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

test("upload retries keep the captured model, snapshot and request identity", async () => {
  const ui = await workspace();
  await flush();
  const file = {name: "input.json", size: 24, text: async () => '{"schema_version":"2.1"}'};
  void ui.get("export-file").emit("change", {target: {files: [file]}});
  await flush();
  ui.pending("/api/v1/validate").reply({step_count: 12, instrument_ids: ["BTC/USD"]});
  await flush();
  ui.get("upload-model").value = "baseline-1";
  await ui.get("upload-model").emit("change");
  void ui.get("classify").click();
  assert.equal(ui.get("upload-model").disabled, true);
  ui.get("upload-model").value = "tcn-1";
  ui.pending("/api/v1/snapshots").reply({snapshot_id: id(2000)});
  await flush();
  const original = ui.pending("/api/v1/analyses");
  assert.deepEqual(JSON.parse(original.options.body), {snapshot_id: id(2000), model_version_id: "baseline-1"});
  original.reply({error: {message: "Connection lost after submission."}}, 503);
  await flush();
  void ui.get("retry").click();
  const repeated = ui.pending("/api/v1/analyses");
  assert.equal(repeated.options.body, original.options.body);
  assert.equal(repeated.options.headers["Idempotency-Key"], original.options.headers["Idempotency-Key"]);
  assert.equal(ui.calls.filter((call) => call.url === "/api/v1/snapshots").length, 1);
});

test("saved experiments rerun without upload and deliberate runs have distinct identities", async () => {
  const ui = await workspace();
  await flush();
  void ui.select(0);
  ui.pending(`/api/v1/analyses/${id(1)}`).reply(metadata("job-A", "failed"));
  await flush();
  assert.equal(ui.get("rerun-controls").hidden, false);
  ui.get("rerun-model").value = "baseline-1";
  await ui.get("rerun-model").emit("change");
  void ui.get("rerun").click();
  const first = ui.pending("/api/v1/analyses");
  assert.deepEqual(JSON.parse(first.options.body), {snapshot_id: id(1000), model_version_id: "baseline-1"});
  assert.equal(ui.calls.filter((call) => call.url === "/api/v1/snapshots").length, 0);
  first.reply({analysis_id: id(3000)}, 202);
  await flush();
  ui.pending(`/api/v1/analyses/${id(3000)}`).reply(metadata("new-job", "queued"));
  await flush();
  void ui.select(0);
  ui.pending(`/api/v1/analyses/${id(1)}`).reply(metadata("job-A", "failed"));
  await flush();
  void ui.get("rerun").click();
  const second = ui.pending("/api/v1/analyses");
  assert.notEqual(first.options.headers["Idempotency-Key"], second.options.headers["Idempotency-Key"]);
});

test("the active model is captured as an immutable version before submission", async () => {
  const ui = await workspace();
  await flush();
  void ui.select(0);
  ui.pending(`/api/v1/analyses/${id(1)}`).reply(metadata("job-A", "failed"));
  await flush();
  ui.get("rerun-model").value = "active";
  void ui.get("rerun").click();
  const submission = ui.pending("/api/v1/analyses");
  assert.equal(JSON.parse(submission.options.body).model_version_id, "tcn-1");
  submission.reply({error: {message: "Response unavailable"}}, 503);
  await flush();
  ui.state.activeModelId = "baseline-1";
  await ui.get("models-refresh").click();
  void ui.get("retry").click();
  const repeated = ui.pending("/api/v1/analyses");
  assert.equal(repeated.options.body, submission.options.body);
  assert.equal(repeated.options.headers["Idempotency-Key"], submission.options.headers["Idempotency-Key"]);
});

test("an old rerun response cannot replace another selected analysis", async () => {
  const ui = await workspace();
  await flush();
  void ui.select(0);
  ui.pending(`/api/v1/analyses/${id(1)}`).reply(metadata("job-A", "failed"));
  await flush();
  void ui.get("rerun").click();
  const submission = ui.pending("/api/v1/analyses");
  void ui.select(1);
  ui.pending(`/api/v1/analyses/${id(2)}`).reply(metadata("job-B", "running"));
  await flush();
  submission.reply({analysis_id: id(3000)}, 202);
  await flush();
  assert.match(ui.get("analysis-identity").textContent, new RegExp(`^${id(2)}`));
  assert.equal(ui.calls.some((call) => call.url === `/api/v1/analyses/${id(3000)}`), false);
});

test("model paging is bounded and preserves a version selected on another page", async () => {
  const ui = await workspace();
  await flush();
  ui.models.splice(0, ui.models.length, ...Array.from({length: 120}, (_, index) => ({model_version_id: `model-${index}`})));
  ui.state.activeModelId = null;
  await ui.get("models-refresh").click();
  assert.equal(ui.get("upload-model").value, "");
  assert.equal(ui.get("models-next").hidden, false);
  ui.get("upload-model").value = "model-9";
  await ui.get("upload-model").emit("change");
  await ui.get("models-next").click();
  assert.equal(ui.get("upload-model").value, "model-9");
  assert.equal(ui.get("upload-model").children.length, 22);
  assert.equal(ui.get("models-next").hidden, true);
  assert.ok(ui.calls.some((call) => call.url === "/api/v1/models?published_only=true&limit=100&offset=100"));
});

test("comparisons page only completed matching snapshots and clear when selection changes", async () => {
  const ui = await workspace(30);
  await flush();
  for (const item of ui.items) {
    item.job.state = "succeeded";
    item.result_available = true;
    item.snapshot_id = id(1000);
  }
  void ui.select(0);
  ui.pending(`/api/v1/analyses/${id(1)}`).reply(metadata("job-A", "succeeded"));
  await flush();
  ui.pending(`/api/v1/analyses/${id(1)}/result`).reply({instruments: [{instrument_id: "BTC/USD"}], time_grid: {step_count: 12}});
  await flush();
  ui.get("comparison-controls").open = true;
  await ui.get("comparison-controls").emit("toggle");
  assert.equal(ui.get("comparison-target").children.length, 20);
  assert.equal(ui.get("comparison-target").children.some((item) => item.value === id(1)), false);
  await ui.get("comparison-next").click();
  assert.equal(ui.get("comparison-target").children.length, 11);
  ui.get("comparison-target").value = id(21);
  await ui.get("comparison-target").emit("change");
  await ui.get("compare").click();
  assert.equal(ui.get("comparison-preview").src, `/api/v1/analyses/${id(1)}/comparison/preview?other=${id(21)}`);
  assert.equal(ui.get("download-comparison").href, `/api/v1/analyses/${id(1)}/comparison?other=${id(21)}`);
  assert.equal(ui.get("report-preview").hidden, true);
  void ui.select(1);
  assert.equal(ui.get("comparison-preview").src, undefined);
  assert.equal(ui.get("comparison-actions").hidden, true);
});

test("late comparison lists cannot expose candidates for an earlier selection", async () => {
  const ui = await workspace();
  await flush();
  void ui.select(0);
  ui.pending(`/api/v1/analyses/${id(1)}`).reply(metadata("job-A", "succeeded"));
  await flush();
  ui.pending(`/api/v1/analyses/${id(1)}/result`).reply({instruments: [], time_grid: {step_count: 12}});
  await flush();
  ui.state.holdHistory = true;
  ui.get("comparison-controls").open = true;
  void ui.get("comparison-controls").emit("toggle");
  const stale = ui.pending(`/api/v1/analyses?snapshot_id=${id(1000)}&state=succeeded&limit=21&offset=0`);
  void ui.select(1);
  stale.reply({items: [{...ui.items[0], snapshot_id: id(1000), result_available: true, job: {...ui.items[0].job, state: "succeeded"}}]});
  await flush();
  assert.equal(ui.get("comparison-target").children.length, 0);
  assert.equal(ui.get("compare").disabled, true);
});

function queuedMetadata() {
  const value = metadata("queued-job", "queued");
  value.job.stage = "queued";
  value.job.created_at = "2026-01-01T00:00:00Z";
  value.job.updated_at = "2026-01-01T00:00:10Z";
  return value;
}

test("queued analyses show worker availability and elapsed wait with bounded checks", async () => {
  const ui = await workspace();
  await flush();
  void ui.select(0);
  ui.pending(`/api/v1/analyses/${id(1)}`).reply(queuedMetadata());
  await flush();
  assert.match(ui.get("job-detail").textContent, /Waiting to start · 20s/);
  ui.pending("/api/v1/service-status").reply({state: "busy", message: "Workers are processing other tasks."});
  await flush();
  assert.match(ui.get("job-detail").textContent, /Workers are processing other tasks/);
  ui.state.now += 1500;
  void ui.timers.values().next().value();
  ui.pending(`/api/v1/analyses/${id(1)}`).reply(queuedMetadata());
  await flush();
  assert.equal(ui.calls.filter((call) => call.url === "/api/v1/service-status").length, 1);
  ui.state.now += 5000;
  void ui.timers.values().next().value();
  ui.pending(`/api/v1/analyses/${id(1)}`).reply(queuedMetadata());
  await flush();
  assert.equal(ui.calls.filter((call) => call.url === "/api/v1/service-status").length, 2);
  ui.pending("/api/v1/service-status").reply({state: "unavailable", message: "No worker has a recent heartbeat."});
  await flush();
  assert.match(ui.get("job-detail").textContent, /26s.*No worker has a recent heartbeat/);
});

test("diagnostic failure does not interrupt polling or loading a completed result", async () => {
  const ui = await workspace();
  await flush();
  void ui.select(0);
  ui.pending(`/api/v1/analyses/${id(1)}`).reply(queuedMetadata());
  await flush();
  ui.pending("/api/v1/service-status").reply({error: {message: "Internal diagnostic failure"}}, 503);
  await flush();
  assert.match(ui.get("job-detail").textContent, /Worker availability could not be checked/);
  assert.equal(ui.get("notice").hidden, true);
  assert.equal(ui.get("retry").hidden, true);
  void ui.timers.values().next().value();
  ui.pending(`/api/v1/analyses/${id(1)}`).reply(metadata("queued-job", "succeeded"));
  await flush();
  ui.pending(`/api/v1/analyses/${id(1)}/result`).reply({instruments: [], time_grid: {step_count: 12}});
  await flush();
  assert.equal(ui.get("job-detail").textContent, "");
  assert.equal(ui.get("result").hidden, false);
  assert.equal(ui.get("report-preview").hidden, false);
});

test("an outstanding diagnostic cannot overwrite a running analysis or a new selection", async () => {
  const ui = await workspace();
  await flush();
  void ui.select(0);
  ui.pending(`/api/v1/analyses/${id(1)}`).reply(queuedMetadata());
  await flush();
  const status = ui.pending("/api/v1/service-status");
  void ui.timers.values().next().value();
  ui.pending(`/api/v1/analyses/${id(1)}`).reply(metadata("queued-job", "running"));
  await flush();
  assert.equal(ui.get("job-detail").textContent, "Analyzing prices.");
  void ui.select(1);
  ui.pending(`/api/v1/analyses/${id(2)}`).reply(metadata("other-job", "running"));
  await flush();
  status.reply({state: "unavailable", message: "No worker has a recent heartbeat."});
  await flush();
  assert.equal(ui.get("job-detail").textContent, "Analyzing prices.");
  assert.match(ui.get("analysis-identity").textContent, new RegExp(`^${id(2)}`));
});

test("an earlier session's diagnostics cannot overwrite a new session", async () => {
  const ui = await workspace();
  await flush();
  void ui.select(0);
  ui.pending(`/api/v1/analyses/${id(1)}`).reply(queuedMetadata());
  await flush();
  const stale = ui.pending("/api/v1/service-status");
  await ui.get("logout").click();
  await ui.get("login-form").emit("submit", {preventDefault() {}});
  void ui.select(1);
  ui.pending(`/api/v1/analyses/${id(2)}`).reply(queuedMetadata());
  await flush();
  stale.reply({state: "unavailable", message: "Old session status"});
  await flush();
  assert.equal(ui.calls.filter((call) => call.url === "/api/v1/service-status").length, 2);
  assert.doesNotMatch(ui.get("job-detail").textContent, /Old session status/);
  assert.match(ui.get("job-detail").textContent, /Checking worker availability/);
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
