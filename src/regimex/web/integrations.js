import {$, element} from "./ui.js";
import {clearStorage, renderStorage, storageError} from "./storage.js";

const ROOT = "/api/v1/admin/integrations";
const PAGE_SIZE = 20;
const scopes = ["read", "upload", "analyze", "delete"];
const eventNames = {
  "integration.created": "Connection added",
  "integration.updated": "Connection updated",
  "key.created": "Service key created",
  "key.revoked": "Service key revoked",
};

function date(value, fallback = "Never") {
  return value ? new Date(value).toLocaleString() : fallback;
}

export function createIntegrations({session, navigate}) {
  let opened = false;
  let selectedId = null;
  let selection = null;
  let generation = 0;
  let storageRequest = 0;
  let mutation = 0;
  let busy = false;
  let detailLoading = false;
  const pages = {
    connections: {
      prefix: "connections", list: "connection-list", label: "connections", empty: "No connections yet.",
      url: () => ROOT, render: renderConnection,
    },
    keys: {
      prefix: "connection-keys", list: "connection-keys", label: "service keys", empty: "No service keys.",
      url: () => `${ROOT}/${selection.integration_id}/keys`, render: renderKey,
    },
    events: {
      prefix: "connection-events", list: "connection-events", label: "access changes", empty: "No access changes.",
      url: () => `${ROOT}/${selection.integration_id}/events`, render: renderEvent,
    },
  };
  for (const page of Object.values(pages)) page.request = page.offset = 0;

  function message(text = "", error = false) {
    $("connection-notice").textContent = text;
    $("connection-notice").hidden = !text;
    $("connection-notice").classList.toggle("error", error);
  }

  function editStatus(text = "") {
    $("connection-edit-status").textContent = text;
    $("connection-edit-status").hidden = !text;
    $("connection-use-saved-name").hidden = !text || $("connection-name").value === selection?.display_name;
  }

  function clearSecret() {
    $("connection-secret-value").value = "";
    $("connection-secret-value").type = "password";
    $("connection-secret-reveal").textContent = "Show key";
    $("connection-secret-label").textContent = "New service key";
    $("connection-issued-id").textContent = "";
    $("connection-copy-status").textContent = "";
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
      for (const input of $(form).querySelectorAll("input, button")) input.disabled = busy || detailLoading;
    }
    $("connection-refresh").disabled = $("connection-toggle").disabled = busy || detailLoading || !selection;
    for (const page of Object.values(pages)) {
      $(`${page.prefix}-pages`).hidden = page.offset === 0 && !page.hasMore;
      $(`${page.prefix}-previous`).disabled = busy || page.loading || page.offset === 0;
      $(`${page.prefix}-next`).disabled = busy || page.loading || !page.hasMore;
    }
    for (const button of $("connection-keys").querySelectorAll("button")) button.disabled = busy || detailLoading;
  }

  function reset() {
    opened = false;
    selectedId = null;
    selection = null;
    generation++;
    mutation++;
    busy = detailLoading = false;
    clearSecret();
    clearStorage("connection-storage");
    message();
    editStatus();
    $("connections").hidden = true;
    $("connection-detail").hidden = true;
    for (const id of ["connection-create-details", "connection-key-details", "connection-setup"]) $(id).open = false;
    for (const id of ["connection-owner", "connection-source", "connection-id", "connection-verify-url",
      "connection-title", "connection-state"]) $(id).textContent = "";
    for (const id of ["connection-name", "connection-new-name", "connection-new-owner",
      "connection-new-source", "connection-key-name"]) $(id).value = "";
    $("connection-key-days").value = "30";
    for (const scope of scopes) $(`connection-scope-${scope}`).checked = scope !== "delete";
    for (const page of Object.values(pages)) {
      page.offset = 0;
      page.hasMore = page.loading = false;
      page.request++;
      $(page.list).replaceChildren();
      $(page.list).setAttribute("aria-busy", "false");
      $(`${page.prefix}-status`).hidden = true;
      $(`${page.prefix}-range`).textContent = "";
    }
  }

  function renderSelection(next, {nameSaved = false} = {}) {
    const name = $("connection-name");
    const hasDraft = selection && name.value !== selection.display_name;
    const renamed = selection && selection.display_name !== next.display_name;
    selection = next;
    $("connection-detail").hidden = false;
    $("connection-title").textContent = selection.display_name;
    if (!hasDraft || nameSaved) name.value = selection.display_name;
    if (name.value === selection.display_name) editStatus();
    else if (renamed)
      editStatus(`The saved name is now “${selection.display_name}”. Review your entry before saving.`);
    $("connection-state").textContent = selection.enabled ? "Active" : "Paused";
    $("connection-owner").textContent = selection.owner_id;
    $("connection-source").textContent = selection.source_namespace;
    $("connection-id").textContent = selection.integration_id;
    $("connection-verify-url").textContent = location.origin + "/api/v1/connection";
    $("connection-toggle").textContent = selection.enabled ? "Pause connection" : "Resume connection";
    $("connection-key-details").hidden = !selection.enabled;
    $("connection-paused").hidden = selection.enabled;
  }

  async function loadPage(name, offset = pages[name].offset) {
    const stamp = mark();
    if (!current(stamp)) return;
    const page = pages[name];
    const request = ++page.request;
    const active = () => current(stamp) && request === page.request;
    const status = $(`${page.prefix}-status`);
    page.loading = true;
    $(page.list).setAttribute("aria-busy", "true");
    status.classList.toggle("error", false);
    status.textContent = `Loading ${page.label}…`;
    status.hidden = false;
    controls();
    try {
      const data = await session.json(`${page.url()}?limit=${PAGE_SIZE + 1}&offset=${offset}`);
      if (!active()) return;
      const items = data.items.slice(0, PAGE_SIZE);
      $(page.list).replaceChildren(...items.map(page.render));
      page.offset = offset;
      page.hasMore = data.items.length > PAGE_SIZE;
      $(`${page.prefix}-range`).textContent = items.length ? `${offset + 1}–${offset + items.length}` : "";
      status.textContent = offset ? `No ${page.label} on this page.` : page.empty;
      status.hidden = items.length !== 0;
    } catch (error) {
      if (!active()) return;
      status.textContent = `Could not load ${page.label}. ${error.message}`;
      status.classList.toggle("error", true);
    } finally {
      if (active()) {
        page.loading = false;
        $(page.list).setAttribute("aria-busy", "false");
        controls();
      }
    }
  }

  function renderConnection(item) {
    const row = element("li", "");
    const link = element("a", "", "button quiet");
    link.href = "/connections/" + encodeURIComponent(item.integration_id);
    link.setAttribute("data-workspace-link", "");
    if (item.integration_id === selectedId) link.setAttribute("aria-current", "page");
    link.append(element("strong", item.display_name), element("span", item.enabled ? "Active" : "Paused", "hint"),
      element("small", `${item.owner_id} / ${item.source_namespace}`));
    row.append(link);
    return row;
  }

  function renderKey(key) {
    const row = element("li", "");
    let state = selection.enabled ? "Active" : "Paused";
    if (key.revoked_at) state = "Revoked";
    else if (key.expires_at && Date.parse(key.expires_at) <= Date.now()) state = "Expired";
    row.append(element("strong", key.name), element("p", `${state} · ${key.scopes.join(", ")}`));
    row.append(element("p", `Created: ${date(key.created_at, "Unknown")} · Expires: ${date(key.expires_at, "No expiry")} · Last used: ${date(key.last_used_at)}`, "hint"));
    row.append(element("p", `Key ID ${key.token_id}`, "identity"));
    if (!key.revoked_at) {
      const button = element("button", "Revoke", "quiet");
      button.type = "button";
      button.addEventListener("click", () => {
        if (!window.confirm(`Revoke “${key.name}” (${key.token_id})? Requests using this key will be rejected.`)) return;
        void write(async (stamp) => {
          const id = selection.integration_id;
          await session.request(`${ROOT}/${id}/keys/${key.token_id}/revoke`, {method: "POST"});
          if (!current(stamp)) return;
          clearSecret();
          await Promise.all([loadPage("keys"), loadPage("events", 0)]);
          if (current(stamp)) message("Service key revoked.");
        });
      });
      row.append(button);
    }
    return row;
  }

  function renderEvent(event) {
    const row = element("li", "");
    row.append(element("strong", eventNames[event.action] || event.action));
    if (event.action.startsWith("integration."))
      row.append(element("p", `${event.display_name} · ${event.enabled ? "Active" : "Paused"}`));
    row.append(element("p", `${date(event.created_at)} · ${event.actor_owner_id || "System"}`, "hint"));
    if (event.target_token_id) row.append(element("p", event.target_token_id, "identity"));
    return row;
  }

  async function loadDetails(id, focus) {
    if (detailLoading) return;
    const stamp = mark();
    detailLoading = true;
    controls();
    message("Loading connection…");
    try {
      if (!await readSelection(id, stamp)) return;
      message();
      if (focus && document.activeElement === $("connections-title")) $("connection-title").focus();
    } catch (error) {
      if (current(stamp)) message(`Could not load this connection. ${error.message} Use Refresh to try again.`, true);
      return;
    } finally {
      if (current(stamp)) {
        detailLoading = false;
        controls();
      }
    }
    await Promise.all([loadPage("keys"), loadPage("events"), loadStorage()]);
  }

  async function readSelection(id, stamp) {
    const saved = await session.json(`${ROOT}/${encodeURIComponent(id)}`);
    if (!current(stamp)) return false;
    renderSelection(saved);
    return true;
  }

  async function loadStorage() {
    const stamp = mark();
    const id = selection.integration_id;
    const request = ++storageRequest;
    try {
      const usage = await session.json(`${ROOT}/${id}/storage`);
      if (current(stamp) && request === storageRequest) renderStorage("connection-storage", usage);
    } catch {
      if (current(stamp) && request === storageRequest) storageError("connection-storage");
    }
  }

  async function write(action) {
    if (busy || detailLoading || !opened || session.value?.principal.role !== "admin") return;
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

  async function show(id = null) {
    if (session.value?.principal.role !== "admin") return;
    const offset = pages.connections.offset;
    reset();
    pages.connections.offset = offset;
    opened = true;
    selectedId = id;
    $("connections").hidden = false;
    $("connections-title").focus();
    await Promise.all([loadPage("connections"), id && loadDetails(id, true)]);
  }

  $("connections-refresh").addEventListener("click", () => {
    clearSecret();
    void loadPage("connections");
    if (selectedId && !selection) void loadDetails(selectedId, false);
  });
  $("connection-refresh").addEventListener("click", () => {
    if (!selection || busy || detailLoading) return;
    storageRequest++;
    clearSecret();
    clearStorage("connection-storage");
    message();
    void Promise.all([loadPage("connections"), loadDetails(selection.integration_id, false)]);
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
      navigate("/connections/" + encodeURIComponent(result.integration_id));
    });
  });

  async function update(fields, stamp) {
    const id = selection.integration_id;
    let result;
    try {
      result = await session.json(`${ROOT}/${id}`, {
        method: "PATCH", body: {expected_revision: selection.revision, ...fields},
      });
    } catch (error) {
      if (!current(stamp) || error.status !== 409) throw error;
      clearSecret();
      try {
        if (!await readSelection(id, stamp)) return;
      } catch (readError) {
        if (current(stamp)) {
          editStatus();
          message(`The connection changed. Use Refresh connection to see the saved settings. ${readError.message}`, true);
        }
        return;
      }
      const saved = `The connection changed. “${selection.display_name}” is ${selection.enabled ? "active" : "paused"}.`;
      let action = "";
      if (fields.display_name !== undefined && fields.display_name !== selection.display_name)
        action = " Review your entry and choose Save name to apply it.";
      else if (fields.enabled !== undefined && fields.enabled !== selection.enabled)
        action = ` Choose ${fields.enabled ? "Resume" : "Pause"} connection again to confirm.`;
      editStatus(saved + action);
      void Promise.all([loadPage("connections"), loadPage("keys"), loadPage("events", 0)]);
      return;
    }
    if (!current(stamp)) return;
    clearSecret();
    editStatus();
    renderSelection(result, {nameSaved: fields.display_name !== undefined});
    await Promise.all([loadPage("connections"), loadPage("keys"), loadPage("events", 0)]);
  }
  $("connection-edit").addEventListener("submit", (event) => {
    event.preventDefault();
    if (selection) void write((stamp) => update({display_name: $("connection-name").value.trim()}, stamp));
  });
  $("connection-use-saved-name").addEventListener("click", () => {
    if (!selection || busy || detailLoading) return;
    $("connection-name").value = selection.display_name;
    editStatus();
    $("connection-name").focus();
  });
  $("connection-toggle").addEventListener("click", () => {
    if (!selection) return;
    if (selection.enabled && !window.confirm(`Pause “${selection.display_name}”? All its keys will stop accepting new requests.`)) return;
    void write((stamp) => update({enabled: !selection.enabled}, stamp));
  });

  $("connection-key-create").addEventListener("submit", (event) => {
    event.preventDefault();
    if (!selection?.enabled) return;
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
        if (error.status && error.status < 500 && error.status !== 408) throw error;
        if (!current(stamp)) return;
        // A gateway timeout can arrive after the key was issued. Its secret is shown only once.
        message("Key creation could not be confirmed. Check the latest keys and revoke any new key you could not save before creating another.", true);
        await Promise.all([loadPage("keys", 0), loadPage("events", 0)]);
        return;
      }
      if (!current(stamp)) return;
      $("connection-secret-label").textContent = `New service key · ${issued.key.name}`;
      $("connection-issued-id").textContent = `Key ID ${issued.key.token_id}`;
      $("connection-secret-value").value = issued.token;
      $("connection-secret").hidden = false;
      $("connection-key-details").open = false;
      $("connection-key-name").value = "";
      $("connection-scope-delete").checked = false;
      $("connection-secret-copy").focus();
      await Promise.all([loadPage("keys", 0), loadPage("events", 0)]);
    });
  });
  $("connection-secret-clear").addEventListener("click", () => {
    clearSecret();
    $("connection-key-summary").focus();
  });
  $("connection-secret-reveal").addEventListener("click", () => {
    const input = $("connection-secret-value");
    const show = input.type === "password";
    input.type = show ? "text" : "password";
    $("connection-secret-reveal").textContent = show ? "Hide key" : "Show key";
    $("connection-copy-status").textContent = "";
    if (show) {
      input.focus();
      input.select();
    }
  });
  $("connection-secret-copy").addEventListener("click", async () => {
    const stamp = mark();
    const token = $("connection-secret-value").value;
    if (!token) return;
    try {
      await navigator.clipboard.writeText(token);
      if (current(stamp) && $("connection-secret-value").value === token)
        $("connection-copy-status").textContent = "Key copied.";
    } catch {
      if (current(stamp) && $("connection-secret-value").value === token) {
        const input = $("connection-secret-value");
        if (input.type === "text") {
          input.focus();
          input.select();
        }
        $("connection-copy-status").textContent = input.type === "password"
          ? "Clipboard unavailable. Choose Show key, then copy the selected text."
          : "Clipboard unavailable. Copy the selected text.";
      }
    }
  });

  for (const [name, page] of Object.entries(pages)) {
    for (const [direction, change] of [["previous", -PAGE_SIZE], ["next", PAGE_SIZE]]) {
      const button = $(`${page.prefix}-${direction}`);
      button.addEventListener("click", () => {
        if (button.disabled || (name !== "connections" && !selection)) return;
        clearSecret();
        void loadPage(name, page.offset + change);
      });
    }
  }
  window.addEventListener("pagehide", () => {
    generation++;
    clearSecret();
  });
  return {reset, show};
}
