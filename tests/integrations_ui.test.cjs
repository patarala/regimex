"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const root = path.resolve(__dirname, "../src/regimex/web");
const markup = fs.readFileSync(path.join(root, "index.html"), "utf8");
const flush = async () => { for (let i = 0; i < 5; i++) await new Promise(setImmediate); };
const connection = (id) => ({integration_id: id, owner_id: "platform", source_namespace: "market",
  display_name: `Platform ${id}`, enabled: true, revision: 0});

class Element {
  constructor() {
    this.children = [];
    this.events = new Map();
    this.value = "";
    this.hidden = false;
    this.disabled = false;
    this.checked = true;
    this.classList = {toggle() {}};
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(name, value) { this[name] = value; }
  querySelectorAll() { return []; }
  addEventListener(name, callback) { this.events.set(name, callback); }
  fire(name) { return this.events.get(name)?.({preventDefault() {}}); }
  focus() {}
  select() {}
}

async function workspace(role = "admin") {
  const nodes = new Map([...markup.matchAll(/\bid="([^"]+)"/g)].map((match) => [match[1], new Element()]));
  const get = (id) => {
    assert.ok(nodes.has(id), `Unknown DOM ID ${id}`);
    return nodes.get(id);
  };
  const calls = [];
  const copied = [];
  const events = new Map();
  const state = {delayIssue: false, delayStorage: false};
  const session = {
    epoch: 1, value: {principal: {role}},
    json(url, options = {}) {
      return new Promise((resolve, reject) => {
        const call = {url, options, resolve, reject};
        calls.push(call);
        if (options.method === "POST" && url.endsWith("/keys")) {
          if (!state.delayIssue) resolve({key: {token_id: "key-1"}, token: "rgx_once-only-secret"});
        } else if (url.includes("/keys?") || url.includes("/events?")) resolve({items: []});
        else if (url.endsWith("/storage")) {
          if (!state.delayStorage) resolve({used_bytes: 1024, reserved_bytes: 0,
            limit_bytes: 2 * 1024 ** 3, available_bytes: 2 * 1024 ** 3 - 1024});
        }
        else if (url.includes("integrations?")) resolve({items: [connection("one"), connection("two")]});
        else if (url.endsWith("/one")) resolve(connection("one"));
        else if (url.endsWith("/two")) resolve(connection("two"));
        else reject(new Error(`Unexpected request ${url}`));
      });
    },
    async request() { throw new Error("Unexpected mutation"); },
  };
  const context = vm.createContext({
    document: {getElementById: get, createElement: () => new Element()},
    window: {addEventListener: (name, callback) => events.set(name, callback), confirm: () => true},
    navigator: {clipboard: {writeText: async (value) => { copied.push(value); }}},
    Date, console,
  });
  const modules = new Map();
  function load(filename) {
    if (!modules.has(filename)) modules.set(filename, new vm.SourceTextModule(fs.readFileSync(filename, "utf8"), {
      context, identifier: filename,
    }));
    return modules.get(filename);
  }
  const entry = load(path.join(root, "integrations.js"));
  await entry.link((specifier, parent) => load(path.resolve(path.dirname(parent.identifier), specifier)));
  await entry.evaluate();
  const controller = entry.namespace.createIntegrations({session});
  controller.activate();
  async function open(index = 0) {
    get("open-connections").fire("click");
    await flush();
    get("connection-list").children[index].children[0].fire("click");
    await flush();
  }
  function issue() {
    get("connection-key-name").value = "Backend";
    get("connection-key-days").value = "30";
    get("connection-key-create").fire("submit");
  }
  return {get, calls, copied, controller, session, open, issue, state, events};
}

test("ordinary users cannot open connection administration or send management requests", async () => {
  const page = await workspace("user");
  assert.equal(page.get("open-connections").hidden, true);
  page.get("open-connections").fire("click");
  page.issue();
  await flush();
  assert.equal(page.calls.length, 0);
});

test("issued secret can be copied and is cleared when leaving connections", async () => {
  const page = await workspace();
  await page.open();
  page.issue();
  await flush();
  assert.equal(page.get("connection-secret-value").value, "rgx_once-only-secret");
  assert.equal(page.get("connection-secret").hidden, false);
  await page.get("connection-secret-copy").fire("click");
  assert.deepEqual(page.copied, ["rgx_once-only-secret"]);
  page.get("close-connections").fire("click");
  assert.equal(page.get("connection-secret-value").value, "");
  assert.equal(page.get("connections").hidden, true);
});

test("a key response from a previous connection never appears in the current connection", async () => {
  const page = await workspace();
  await page.open();
  page.state.delayIssue = true;
  page.issue();
  await flush();
  const issued = page.calls.find((call) => call.options.method === "POST");
  page.get("connection-list").children[1].children[0].fire("click");
  await flush();
  issued.resolve({key: {}, token: "rgx_old-connection-secret"});
  await flush();
  assert.equal(page.get("connection-title").textContent, "Platform two");
  assert.equal(page.get("connection-secret-value").value, "");
  assert.equal(page.get("connection-secret").hidden, true);
});

test("logout invalidates pending issuance and clears connection details", async () => {
  const page = await workspace();
  await page.open();
  page.state.delayIssue = true;
  page.issue();
  await flush();
  const issued = page.calls.find((call) => call.options.method === "POST");
  page.session.epoch++;
  page.session.value = null;
  page.controller.reset();
  issued.resolve({key: {}, token: "rgx_previous-session-secret"});
  await flush();
  assert.equal(page.get("connection-secret-value").value, "");
  assert.equal(page.get("connection-detail").hidden, true);
  assert.equal(page.get("open-connections").hidden, true);
});

test("pagehide clears the secret and empty permissions do not create a key", async () => {
  const page = await workspace();
  await page.open();
  for (const scope of ["read", "upload", "analyze"]) page.get(`connection-scope-${scope}`).checked = false;
  page.issue();
  await flush();
  assert.equal(page.calls.some((call) => call.options.method === "POST"), false);
  assert.match(page.get("connection-notice").textContent, /permission/);
  page.get("connection-scope-read").checked = true;
  page.issue();
  await flush();
  page.events.get("pagehide")();
  assert.equal(page.get("connection-secret-value").value, "");
});

test("a pending key cannot reappear after leaving the document", async () => {
  const page = await workspace();
  await page.open();
  page.state.delayIssue = true;
  page.issue();
  await flush();
  const issued = page.calls.find((call) => call.options.method === "POST");
  page.events.get("pagehide")();
  issued.resolve({key: {}, token: "rgx_page-that-was-left"});
  await flush();
  assert.equal(page.get("connection-secret-value").value, "");
  assert.equal(page.get("connection-secret").hidden, true);
});

test("connection storage follows selection and is cleared when signing out", async () => {
  const page = await workspace();
  await page.open();
  assert.equal(page.get("connection-storage-usage").textContent, "Storage: 1 KiB of 2 GiB");
  page.state.delayStorage = true;
  page.get("connection-refresh").fire("click");
  await flush();
  const stale = page.calls.filter((call) => call.url.endsWith("/one/storage")).at(-1);
  page.get("connection-list").children[1].children[0].fire("click");
  await flush();
  const latest = page.calls.find((call) => call.url.endsWith("/two/storage"));
  latest.resolve({used_bytes: 2048, reserved_bytes: 1024, limit_bytes: 4096, available_bytes: 1024});
  await flush();
  stale.resolve({used_bytes: 0, reserved_bytes: 0, limit_bytes: 4096, available_bytes: 4096});
  await flush();
  assert.equal(page.get("connection-storage-usage").textContent, "Storage: 3 KiB of 4 KiB");
  assert.match(page.get("connection-storage-reserved").textContent, /1 KiB reserved/);
  page.get("connection-refresh").fire("click");
  await flush();
  const signedOut = page.calls.filter((call) => call.url.endsWith("/two/storage")).at(-1);
  page.session.epoch++;
  page.session.value = null;
  page.controller.reset();
  signedOut.resolve({used_bytes: 4096, reserved_bytes: 0, limit_bytes: 4096, available_bytes: 0});
  await flush();
  assert.equal(page.get("connection-storage-status").hidden, true);
  assert.equal(page.get("connection-storage-usage").textContent, "");
});
