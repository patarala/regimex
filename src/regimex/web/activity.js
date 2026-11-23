import {$, element, shortDate} from "./ui.js";
import {isTransientRequestError} from "./session.js";

const endpoint = "/api/v1/terminal-activities";
const pageSize = 20;
const statuses = {running: "Running", succeeded: "Succeeded", failed: "Failed", interrupted: "Interrupted"};
const pagePath = (offset, id = "") => "/activity" + (id ? "/" + encodeURIComponent(id) : "") +
  (offset ? "?offset=" + offset : "");

function time(value) {
  const node = element("time", shortDate(value));
  node.dateTime = value;
  return node;
}

export function createActivity(session) {
  const panel = $("activity"), list = $("activity-list"), detail = $("activity-detail");
  const title = $("activity-title"), status = $("activity-status");
  let selectedId = null, offset = 0, generation = 0, timer, rendered = null;

  function close() {
    generation++;
    clearTimeout(timer);
    panel.hidden = true;
    detail.hidden = $("activity-pages").hidden = $("activity-empty").hidden = true;
    rendered = null;
    list.replaceChildren();
    $("activity-facts").replaceChildren();
    title.textContent = "Terminal activity";
    $("activity-result").textContent = $("activity-note").textContent = status.textContent = "";
  }

  function showList(items) {
    const rows = items.slice(0, pageSize);
    const signature = JSON.stringify(rows.map(({activity_id, title, status, started_at}) =>
      [activity_id, title, status, started_at]));
    if (signature !== rendered) {
      const focusId = document.activeElement?.dataset.activityId;
      list.replaceChildren(...rows.map(row => {
        const item = element("li"), link = element("a");
        link.href = pagePath(offset, row.activity_id);
        link.dataset.workspaceLink = "";
        link.dataset.activityId = row.activity_id;
        link.append(element("strong", row.title), time(row.started_at), element("span", statuses[row.status]));
        item.append(link);
        return item;
      }));
      if (focusId) Array.from(list.children).find(li => li.firstChild.dataset.activityId === focusId)
        ?.firstChild.focus({preventScroll: true});
      rendered = signature;
    }
    status.textContent = rows.length ? "" : offset ? "No activity on this page." : "No terminal activity recorded yet.";
    $("activity-empty").hidden = Boolean(rows.length || offset);
    const newer = $("activity-newer"), older = $("activity-older");
    newer.href = pagePath(Math.max(0, offset - pageSize));
    older.href = pagePath(offset + pageSize);
    newer.hidden = offset === 0;
    older.hidden = items.length <= pageSize;
    $("activity-pages").hidden = newer.hidden && older.hidden;
    return rows.some(row => row.status === "running");
  }

  function showDetail(row) {
    title.textContent = row.title;
    const entries = [["Status", statuses[row.status]], ["Started", time(row.started_at)]];
    if (row.finished_at) entries.push(["Finished", time(row.finished_at)]);
    if (row.exit_code != null) entries.push(["Exit code", String(row.exit_code)]);
    $("activity-facts").replaceChildren(...entries.flatMap(([name, value]) => {
      const definition = element("dd");
      definition.append(value);
      return [element("dt", name), definition];
    }));
    for (const [id, value] of [["activity-note", row.note], ["activity-result", row.result_text]]) {
      $(id + "-section").hidden = value == null;
      if ($(id).textContent !== (value ?? "")) $(id).textContent = value ?? "";
    }
    detail.hidden = false;
    status.textContent = "";
    return row.status === "running";
  }

  async function refresh() {
    clearTimeout(timer);
    const request = ++generation;
    if (panel.hidden || document.hidden) return;
    let poll = false;
    try {
      const result = await session.json(selectedId ? endpoint + "/" + encodeURIComponent(selectedId)
        : endpoint + "?limit=" + (pageSize + 1) + "&offset=" + offset);
      if (request !== generation) return;
      poll = selectedId ? showDetail(result) : showList(result.items);
    } catch (error) {
      if (request !== generation) return;
      poll = isTransientRequestError(error);
      status.textContent = error.message;
    }
    if (poll) timer = setTimeout(refresh, 4000);
  }

  $("activity-refresh").addEventListener("click", refresh);
  function show(id, rawOffset) {
    close();
    selectedId = id;
    const start = Number(rawOffset);
    offset = Number.isInteger(start) && start >= 0 && start <= 1000000 ? start : 0;
    panel.hidden = false;
    $("activity-example").textContent = 'regimex activity run --origin ' + location.origin +
      ' --token-file TOKEN_FILE --title "…" -- COMMAND';
    list.hidden = Boolean(id);
    $("activity-back").href = pagePath(offset);
    $("activity-back").hidden = !id;
    status.textContent = id ? "Loading activity…" : "Loading terminal activity…";
    title.focus();
    return refresh();
  }
  return {close, refresh, show};
}
