import {$} from "./ui.js";

export function createRunNameEditor({session, getRun, applyName}) {
  const form = $("training-name-editor"), input = $("training-edited-name");
  const rename = $("training-rename"), save = $("training-name-save");
  const status = $("training-name-status");
  let edit = 0, revision = 0;

  function close(focus = false) {
    edit++;
    form.hidden = true;
    rename.hidden = false;
    input.disabled = save.disabled = false;
    input.value = status.textContent = "";
    if (focus) rename.focus();
  }

  rename.addEventListener("click", () => {
    const run = getRun();
    if (!run) return;
    edit++;
    revision = run.name_revision;
    input.value = run.display_name || "";
    form.hidden = false;
    rename.hidden = true;
    input.focus();
  });
  $("training-name-cancel").addEventListener("click", () => close(true));
  form.addEventListener("submit", async event => {
    event.preventDefault();
    if (form.hidden || save.disabled) return;
    const stamp = edit, workspace = session.epoch, id = getRun().run_id;
    const path = "/api/v1/training-runs/" + encodeURIComponent(id);
    const name = input.value.trim() ? input.value : null;
    const current = () => stamp === edit && workspace === session.epoch;
    input.disabled = save.disabled = true;
    status.textContent = "Saving…";
    try {
      const result = await session.json(path + "/name", {method: "PATCH",
        body: {display_name: name, expected_revision: revision}});
      if (workspace === session.epoch) applyName(id, result.display_name, result.revision);
      if (current()) close(true);
    } catch (error) {
      if (!current()) return;
      status.textContent = error.message;
      if (error.status === 409) {
        try {
          const latest = await session.json(path);
          if (!current()) return;
          revision = latest.name_revision;
          applyName(id, latest.display_name, revision);
          if (latest.display_name === (name?.trim().normalize("NFC") || null)) close(true);
          else status.textContent = "The name changed elsewhere. Review the current name above and save again.";
        } catch (refreshError) { if (current()) status.textContent = refreshError.message; }
      }
    } finally {
      if (current()) {
        input.disabled = save.disabled = false;
        input.focus();
      }
    }
  });
  return {close};
}
