import {$, notice} from "./ui.js";

export function canRetryRead(error) {
  return !error.status || error.status === 408 || error.status === 429 || error.status >= 500;
}

function requestError(data) {
  const error = data?.error;
  const message = error?.message || "The request could not be completed.";
  if (error?.code !== "invalid_data" || !Array.isArray(error.details)) return message;
  const details = error.details.slice(0, 5).map((item) => {
    const path = Array.isArray(item.location)
      ? item.location.filter((part) => part !== "body").join(".").slice(0, 200)
      : "";
    return `${path || "Export"}: ${String(item.message || "Invalid value").slice(0, 300)}`;
  });
  if (error.details.length > 5) details.push(`${error.details.length - 5} more validation errors.`);
  return [message, ...details].join("\n");
}

// The epoch invalidates every request started under an earlier session.
export function createSession({onActivate, onReset}) {
  let value = null;
  let epoch = 0;
  function reset() {
    epoch++;
    value = null;
    $("token").value = "";
    $("owner").textContent = "";
    $("workspace").hidden = true;
    $("account").hidden = true;
    $("logout").hidden = true;
    $("login-panel").hidden = false;
    onReset();
  }

  async function request(path, { method = "GET", body = undefined, headers = {}, signal } = {}) {
    const requestEpoch = epoch;
    const options = {
      method,
      credentials: "same-origin",
      redirect: "error",
      cache: "no-store",
      headers: { ...headers },
      signal,
    };
    if (body !== undefined) {
      options.body = typeof body === "string" ? body : JSON.stringify(body);
      options.headers["Content-Type"] = "application/json";
    }
    if (!["GET", "HEAD"].includes(method) && value)
      options.headers["X-CSRF-Token"] = value.csrf_token;
    let response;
    try {
      response = await fetch(path, options);
    } catch {
      throw new Error("Could not connect to the service. Check your connection and try again.");
    }
    if (requestEpoch !== epoch)
      throw new Error("The workspace changed. Refresh the page to continue.");
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      if (requestEpoch !== epoch)
        throw new Error("The workspace changed. Refresh the page to continue.");
      if (response.status === 401 && path !== "/api/v1/session") {
        reset();
        throw new Error("Your session has expired. Sign in to reopen your saved analyses.");
      }
      const error = new Error(requestError(data));
      error.status = response.status;
      throw error;
    }
    return response;
  }

  async function json(path, options) {
    const requestEpoch = epoch;
    const value = await (await request(path, options)).json();
    if (requestEpoch !== epoch)
      throw new Error("The workspace changed. Refresh the page to continue.");
    return value;
  }

  async function activate(next) {
    epoch++;
    value = next;
    $("token").value = "";
    $("owner").textContent = next.principal.owner_id;
    $("login-panel").hidden = true;
    $("workspace").hidden = false;
    $("account").hidden = false;
    $("logout").hidden = false;
    notice();
    await onActivate();
  }
  $("login-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    $("login-submit").disabled = true;
    const token = $("token").value.trim();
    $("token").value = "";
    try {
      await activate(await json("/api/v1/session", { method: "POST", body: { token } }));
    } catch (error) {
      notice(error.message, true);
    } finally {
      $("login-submit").disabled = false;
    }
  });
  $("logout").addEventListener("click", async () => {
    const workspace = epoch;
    try {
      await request("/api/v1/session", { method: "DELETE" });
    } catch (error) {
      if (workspace !== epoch) return;
      if (error.status !== 401) {
        notice(error.message, true);
        return;
      }
    }
    reset();
    window.history.replaceState(null, "", new URL("/", location.href));
    notice();
  });
  async function start() {
    const workspace = epoch;
    let next;
    try {
      next = await json("/api/v1/session");
    } catch (error) {
      if (workspace !== epoch) return;
      reset();
      if (error.status !== 401) notice(error.message, true);
      return;
    }
    await activate(next);
  }
  window.addEventListener("pagehide", (event) => { if (event.persisted) reset(); });
  window.addEventListener("pageshow", (event) => { if (event.persisted) void start(); });
  return {request, json, start, get value() { return value; }, get epoch() { return epoch; }};
}
