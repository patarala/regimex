import {$} from "./ui.js";

function bytes(value) {
  if (value < 1024) return `${value} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let size = value / 1024;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index++;
  }
  return `${size.toLocaleString(undefined, {maximumFractionDigits: 1})} ${units[index]}`;
}

export function clearStorage(prefix) {
  $(`${prefix}-status`).hidden = true;
  for (const name of ["usage", "available", "reserved"])
    $(`${prefix}-${name}`).textContent = "";
  $(`${prefix}-reserved`).hidden = true;
}

export function renderStorage(prefix, usage) {
  $(`${prefix}-status`).hidden = false;
  $(`${prefix}-usage`).textContent =
    `Storage: ${bytes(usage.used_bytes + usage.reserved_bytes)} of ${bytes(usage.limit_bytes)}`;
  $(`${prefix}-available`).textContent = usage.available_bytes > 0
    ? `${bytes(usage.available_bytes)} available`
    : "Storage limit reached. Contact your administrator before adding data.";
  $(`${prefix}-reserved`).hidden = usage.reserved_bytes === 0;
  $(`${prefix}-reserved`).textContent = usage.reserved_bytes > 0
    ? `${bytes(usage.used_bytes)} saved · ${bytes(usage.reserved_bytes)} reserved for uploads or cleanup.` : "";
}

export function storageError(prefix) {
  clearStorage(prefix);
  $(`${prefix}-status`).hidden = false;
  $(`${prefix}-usage`).textContent = "Storage usage unavailable.";
  $(`${prefix}-available`).textContent = "Refresh to try again.";
}

export function createStorage({session}) {
  let request = 0;

  function reset() {
    request++;
    clearStorage("storage");
  }

  async function load() {
    if (!session.value) return;
    const epoch = session.epoch;
    const serial = ++request;
    const current = () => session.value && epoch === session.epoch && serial === request;
    try {
      const usage = await session.json("/api/v1/storage");
      if (current()) renderStorage("storage", usage);
    } catch {
      if (current()) storageError("storage");
    }
  }

  $("storage-refresh").addEventListener("click", () => { void load(); });
  return {load, reset};
}
