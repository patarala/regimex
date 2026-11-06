import {$} from "./ui.js";
import {clearStorage, renderStorage, storageError} from "./storage.js";

const ROOT = "/api/v1/admin/integrations";
const PAGE_SIZE = 20;
const scopes = ["read", "upload", "analyze"];
const eventNames = {
  "integration.created": "Connection added",
  "integration.updated": "Connection updated",
  "key.created": "Service key created",
  "key.revoked": "Service key revoked",
};

function element(tag, text, className = "") {
  const node = document.createElement(tag);
  node.textContent = text;
  node.className = className;
  return node;
}

function date(value, fallback = "Never") {
  return value ? new Date(value).toLocaleString() : fallback;
}

export function createIntegrations({session}) {
  let opened = false;
  let selection = null;
  let generation = 0;
  let mutation = 0;
  let busy = false;
  const pages = {
    connections: {offset: 0, count: 0, request: 0},
    keys: {offset: 0, count: 0, request: 0},
    events: {offset: 0, count: 0, request: 0},
  };

  function message(text = "", error = false) {
    $("connection-notice").textContent = text;
    $("connection-notice").hidden = !text;
    $("connection-notice").classList.toggle("error", error);
  }

  function clearSecret() {
    $("connection-secret-value").value = "";
    $("connection-secret").hidden = true;
  }

  function mark() {
    return {epoch: session.epoch, generation};
  }

  function current(stamp) {
    return opened && session.value?.principal.role === "admin"
      && stamp.epoch === session.epoch && stamp.generation === generation;
  }

  function controls() {
    for (const form of ["connection-create", "connection-edit", "connection-key-create"]) {
      for (const input of $(form).querySelectorAll("input, button")) input.disabled = busy;
    }
    $("connection-toggle").disabled = busy || !selection;
    for (const [name, page] of Object.entries(pages)) {
      const prefix = name === "connections" ? "connections" : `connection-${name}`;
      $(`${prefix}-pages`).hidden = page.offset === 0 && page.count < PAGE_SIZE;
      $(`${prefix}-previous`).disabled = busy || page.offset === 0;
      $(`${prefix}-next`).disabled = busy || page.count < PAGE_SIZE;
    }
    for (const button of $("connection-keys").querySelectorAll("button")) button.disabled = busy;
  }

  function reset() {
    opened = false;
    selection = null;
    generation++;
    mutation++;
    busy = false;
    clearSecret();
    clearStorage("connection-storage");
    message();
    $("open-connections").hidden = true;
    $("connections").hidden = true;
    $("connection-detail").hidden = true;
    for (const id of ["connection-list", "connection-keys", "connection-events"])
      $(id).replaceChildren();
    for (const id of ["connection-name", "connection-new-name", "connection-new-owner",
      "connection-new-source", "connection-key-name"]) $(id).value = "";
    $("connection-key-days").value = "30";
    for (const scope of scopes) $(`connection-scope-${scope}`).checked = true;
    for (const page of Object.values(pages)) {
      page.offset = page.count = 0;
      page.request++;
    }
  }

  function activate() {
    reset();
    $("open-connections").hidden = session.value?.principal.role !== "admin";
  }

  function renderSelection() {
    $("connection-detail").hidden = !selection;
    if (!selection) return;
    $("connection-title").textContent = selection.display_name;
    $("connection-name").value = selection.display_name;
    $("connection-state").textContent = selection.enabled ? "Active" : "Paused";
    $("connection-identity").textContent =
      `${selection.source_namespace} · ${selection.owner_id}`;
    $("connection-toggle").textContent = selection.enabled ? "Pause connection" : "Resume connection";
    $("connection-key-details").hidden = !selection.enabled;
  }

  async function readPage(name, url) {
    const stamp = mark();
    const page = pages[name];
    const request = ++page.request;
    try {
      const result = await session.json(`${url}?limit=${PAGE_SIZE}&offset=${page.offset}`);
      return current(stamp) && request === page.request ? result : null;
    } catch (error) {
      if (current(stamp) && request === page.request) throw error;
      return null;
    }
  }

  async function loadList() {
    const result = await readPage("connections", ROOT);
    if (!result) return;
    const page = pages.connections;
    page.count = result.items.length;
    $("connection-list").replaceChildren(...result.items.map((item) => {
      const row = element("li", "");
      const button = element("button", "", "quiet");
      button.type = "button";
      button.setAttribute("aria-current", String(item.integration_id === selection?.integration_id));
      button.append(element("span", item.display_name), element("span", item.enabled ? "Active" : "Paused", "hint"));
      button.addEventListener("click", () => runRead(() => select(item.integration_id)));
      row.append(button);
      return row;
    }));
    $("connections-empty").hidden = result.items.length !== 0;
    controls();
  }

  function renderKey(key) {
    const row = element("li", "");
    let state = selection.enabled ? "Active" : "Paused";
    if (key.revoked_at) state = "Revoked";
    else if (key.expires_at && Date.parse(key.expires_at) <= Date.now()) state = "Expired";
    row.append(element("strong", key.name), element("p", `${state} · ${key.scopes.join(", ")}`));
    row.append(element("p", `Expires: ${date(key.expires_at, "No expiry")} · Last used: ${date(key.last_used_at)}`, "hint"));
    row.append(element("p", key.token_id, "identity"));
    if (!key.revoked_at) {
      const button = element("button", "Revoke", "quiet");
      button.type = "button";
      button.addEventListener("click", () => {
        if (!window.confirm(`Revoke “${key.name}”? Requests using this key will be rejected.`)) return;
        void write(async (stamp) => {
          const id = selection.integration_id;
          await session.request(`${ROOT}/${id}/keys/${key.token_id}/revoke`, {method: "POST"});
          if (!current(stamp)) return;
          clearSecret();
          await Promise.all([loadKeys(), loadEvents()]);
          if (current(stamp)) message("Service key revoked.");
        });
      });
      row.append(button);
    }
    return row;
  }

  async function loadKeys() {
    const result = await readPage("keys", `${ROOT}/${selection.integration_id}/keys`);
    if (!result) return;
    const page = pages.keys;
    page.count = result.items.length;
    $("connection-keys").replaceChildren(...result.items.map(renderKey));
    $("connection-keys-empty").hidden = result.items.length !== 0;
    controls();
  }

  async function loadEvents() {
    const result = await readPage("events", `${ROOT}/${selection.integration_id}/events`);
    if (!result) return;
    const page = pages.events;
    page.count = result.items.length;
    $("connection-events").replaceChildren(...result.items.map((event) => {
      const row = element("li", "");
      row.append(element("strong", eventNames[event.action] || event.action));
      if (event.action.startsWith("integration."))
        row.append(element("p", `${event.display_name} · ${event.enabled ? "Active" : "Paused"}`));
      row.append(element("p", `${date(event.created_at)} · ${event.actor_owner_id || "System"}`, "hint"));
      if (event.target_token_id) row.append(element("p", event.target_token_id, "identity"));
      return row;
    }));
    $("connection-events-empty").hidden = result.items.length !== 0;
    controls();
  }

  async function select(id) {
    generation++;
    clearSecret();
    clearStorage("connection-storage");
    message();
    selection = null;
    $("connection-detail").hidden = true;
    $("connection-keys").replaceChildren();
    $("connection-events").replaceChildren();
    pages.keys.offset = pages.events.offset = 0;
    const stamp = mark();
    const result = await session.json(`${ROOT}/${id}`);
    if (!current(stamp)) return;
    selection = result;
    renderSelection();
    await Promise.all([loadKeys(), loadEvents(), loadList(), loadStorage()]);
  }

  async function loadStorage() {
    const stamp = mark();
    const id = selection.integration_id;
    try {
      const usage = await session.json(`${ROOT}/${id}/storage`);
      if (current(stamp)) renderStorage("connection-storage", usage);
    } catch {
      if (current(stamp)) storageError("connection-storage");
    }
  }

  async function runRead(action) {
    let stamp = mark();
    try {
      const result = action();
      stamp = mark();
      await result;
    } catch (error) {
      if (current(stamp)) message(error.message, true);
    }
  }

  async function write(action) {
    if (busy || !opened || session.value?.principal.role !== "admin") return;
    const task = ++mutation;
    const stamp = mark();
    busy = true;
    message();
    controls();
    try {
      await action(stamp);
    } catch (error) {
      if (current(stamp)) message(error.message, true);
    } finally {
      if (task === mutation) {
        busy = false;
        if (opened) controls();
      }
    }
  }

  $("open-connections").addEventListener("click", () => {
    if (session.value?.principal.role !== "admin") return;
    opened = true;
    generation++;
    clearSecret();
    $("workspace").hidden = true;
    $("connections").hidden = false;
    void runRead(loadList);
  });
  $("close-connections").addEventListener("click", () => {
    reset();
    $("open-connections").hidden = session.value?.principal.role !== "admin";
    $("workspace").hidden = !session.value;
  });
  $("connections-refresh").addEventListener("click", () => {
    clearSecret();
    void runRead(loadList);
  });
  $("connection-refresh").addEventListener("click", () => {
    if (selection) void runRead(() => select(selection.integration_id));
  });

  $("connection-create").addEventListener("submit", (event) => {
    event.preventDefault();
    void write(async (stamp) => {
      const result = await session.json(ROOT, {method: "POST", body: {
        display_name: $("connection-new-name").value.trim(),
        owner_id: $("connection-new-owner").value.trim(),
        source_namespace: $("connection-new-source").value.trim(),
      }});
      if (!current(stamp)) return;
      $("connection-create-details").open = false;
      for (const id of ["connection-new-name", "connection-new-owner", "connection-new-source"])
        $(id).value = "";
      pages.connections.offset = 0;
      await runRead(() => select(result.integration_id));
    });
  });

  async function update(fields, stamp) {
    const result = await session.json(`${ROOT}/${selection.integration_id}`, {
      method: "PATCH", body: {expected_revision: selection.revision, ...fields},
    });
    if (!current(stamp)) return;
    selection = result;
    clearSecret();
    renderSelection();
    await Promise.all([loadList(), loadKeys(), loadEvents()]);
  }
  $("connection-edit").addEventListener("submit", (event) => {
    event.preventDefault();
    if (selection) void write((stamp) => update({display_name: $("connection-name").value.trim()}, stamp));
  });
  $("connection-toggle").addEventListener("click", () => {
    if (!selection) return;
    if (selection.enabled && !window.confirm(`Pause “${selection.display_name}”? All its keys will stop accepting new requests.`)) return;
    void write((stamp) => update({enabled: !selection.enabled}, stamp));
  });

  $("connection-key-create").addEventListener("submit", (event) => {
    event.preventDefault();
    if (!selection) return;
    const selectedScopes = scopes.filter((scope) => $(`connection-scope-${scope}`).checked);
    if (!selectedScopes.length) {
      message("Choose at least one permission.", true);
      return;
    }
    void write(async (stamp) => {
      clearSecret();
      let issued;
      try {
        issued = await session.json(`${ROOT}/${selection.integration_id}/keys`, {
          method: "POST", body: {name: $("connection-key-name").value.trim(),
            lifetime_days: Number($("connection-key-days").value), scopes: selectedScopes},
        });
      } catch (error) {
        if (!error.status) throw new Error("Key creation could not be confirmed. Refresh the key list and revoke any new key you could not save before creating another.");
        throw error;
      }
      if (!current(stamp)) return;
      $("connection-secret-value").value = issued.token;
      $("connection-secret").hidden = false;
      $("connection-key-details").open = false;
      $("connection-key-name").value = "";
      pages.keys.offset = pages.events.offset = 0;
      await Promise.all([loadKeys(), loadEvents()]);
    });
  });
  $("connection-secret-clear").addEventListener("click", clearSecret);
  $("connection-secret-copy").addEventListener("click", async () => {
    const stamp = mark();
    const token = $("connection-secret-value").value;
    if (!token) return;
    try {
      await navigator.clipboard.writeText(token);
      if (current(stamp)) message("Key copied.");
    } catch {
      if (current(stamp)) {
        $("connection-secret-value").focus();
        $("connection-secret-value").select();
        message("Clipboard access is unavailable. Copy the selected key with your keyboard.");
      }
    }
  });

  for (const [name, load] of Object.entries({connections: loadList, keys: loadKeys, events: loadEvents})) {
    const prefix = name === "connections" ? "connections" : `connection-${name}`;
    for (const [direction, change] of [["previous", -PAGE_SIZE], ["next", PAGE_SIZE]]) {
      $(`${prefix}-${direction}`).addEventListener("click", () => {
        if (busy || (name !== "connections" && !selection)) return;
        clearSecret();
        pages[name].offset = Math.max(0, pages[name].offset + change);
        void runRead(load);
      });
    }
  }
  window.addEventListener("pagehide", () => {
    generation++;
    clearSecret();
  });
  return {activate, reset};
}
