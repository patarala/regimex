import {$} from "./ui.js";

export function createTrainingProgress(session, metrics) {
  const notice = $("training-progress-notice");
  let current = null;

  function stop() {
    current?.request?.abort();
    current = null;
    notice.hidden = true;
  }

  async function poll(path, attempt, policy) {
    if (!current || current.attempt !== attempt) {
      stop();
      metrics.clear();
      current = {attempt, cursor: 0, history: {}, request: null};
    }
    const state = current;
    if (state.request) return;
    const request = new AbortController();
    state.request = request;
    const timeout = setTimeout(() => request.abort(), 15000);
    let changed = false;
    try {
      do {
        const url = path + "?attempt=" + attempt + "&after=" + state.cursor;
        const page = await session.json(url, {signal: request.signal});
        if (state !== current) return;
        notice.hidden = true;
        if (page.attempt !== attempt) {
          stop();
          metrics.clear();
          return;
        }
        if (page.cursor <= state.cursor) break;
        const firstPage = state.cursor === 0;
        for (const [key, rows] of Object.entries(page.history)) {
          (state.history[key] ??= []).push(...rows);
        }
        state.cursor = page.cursor;
        changed = true;
        if (!page.has_more || document.hidden) break;
        if (firstPage) {
          metrics.show(state.history, null, policy);
          changed = false;
        }
      } while (state === current);
    } catch {
      if (state === current) notice.hidden = false;
    } finally {
      clearTimeout(timeout);
      state.request = null;
      if (changed && state === current) metrics.show(state.history, null, policy);
    }
  }
  return {stop, poll};
}
