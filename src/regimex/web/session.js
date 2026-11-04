import {$, notice} from "./ui.js";

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
    $("workspace").hidden = true;
    $("account").hidden = true;
    $("login-panel").hidden = false;
    onReset();
  }

  async function request(path, { method = "GET", body = undefined, headers = {} } = {}) {
    const requestEpoch = epoch;
    const options = {
      method,
      credentials: "same-origin",
      redirect: "error",
      cache: "no-store",
      headers: { ...headers },
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
      throw new Error(requestError(data));
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
    value = next;
    $("token").value = "";
    $("owner").textContent = next.principal.owner_id;
    $("login-panel").hidden = true;
    $("workspace").hidden = false;
    $("account").hidden = false;
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
    try {
      await request("/api/v1/session", { method: "DELETE" });
      reset();
      const url = new URL(location.href);
      url.searchParams.delete("analysis");
      window.history.replaceState(null, "", url);
      notice();
    } catch (error) {
      notice(error.message, true);
    }
  });
  function start() {
    return json("/api/v1/session").then(activate).catch((error) => {
      reset();
      if (!error.message.includes("credential") && !error.message.includes("session"))
        notice(error.message, true);
    });
  }
  return {request, json, start, get value() { return value; }, get epoch() { return epoch; }};
}
