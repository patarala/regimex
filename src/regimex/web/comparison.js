import {$, option} from "./ui.js";

const COMPARISON_PAGE_SIZE = 20;

export function createComparison({session, getSelection}) {
  let comparisonGeneration = 0;
  let comparisonOffset = 0;
  let comparisonHasMore = false;
  let comparisonItems = [];
  function reset() {
    clearComparison();
    $("comparison-controls").hidden = true;
    $("comparison-controls").open = false;
    comparisonGeneration++;
    comparisonOffset = 0;
    comparisonHasMore = false;
    comparisonItems = [];
    $("comparison-target").replaceChildren();
    $("comparison-target").disabled = true;
    $("compare").disabled = true;
    $("comparison-status").textContent = "";
  }

  function clearComparison() {
    $("comparison-preview").removeAttribute("src");
    $("comparison-preview").hidden = true;
    $("comparison-actions").hidden = true;
    $("download-comparison").removeAttribute("href");
  }

  async function comparisonPage(offset = 0) {
    const selection = getSelection();
    if (!session.value || selection?.metadata?.job.state !== "succeeded"
      || !selection.metadata.snapshot_id) return;
    const generation = ++comparisonGeneration;
    const workspace = session.epoch;
    const active = () => workspace === session.epoch && getSelection() === selection
      && generation === comparisonGeneration;
    $("comparison-target").disabled = $("compare").disabled = true;
    $("comparison-previous").disabled = $("comparison-next").disabled = true;
    $("comparison-status").textContent = "Loading completed analyses…";
    try {
      const query = new URLSearchParams({
        snapshot_id: selection.metadata.snapshot_id,
        state: "succeeded", limit: COMPARISON_PAGE_SIZE + 1, offset,
      });
      const data = await session.json(`/api/v1/analyses?${query}`);
      if (!active()) return;
      comparisonOffset = offset;
      comparisonHasMore = data.items.length > COMPARISON_PAGE_SIZE;
      comparisonItems = data.items.slice(0, COMPARISON_PAGE_SIZE).filter((item) =>
        item.analysis_id !== selection.id && item.snapshot_id === selection.metadata.snapshot_id
        && item.job.state === "succeeded" && item.result_available);
      const picker = $("comparison-target");
      picker.replaceChildren(option("", "Choose an analysis"));
      comparisonItems.forEach((item) => {
        const date = new Date(item.job.created_at).toLocaleString();
        picker.append(option(item.analysis_id,
          `${item.job.model?.model_version_id || "Model unavailable"} · ${date} · ${item.analysis_id.slice(-8)}`));
      });
      picker.value = "";
      $("comparison-status").textContent = comparisonItems.length ? "" : offset || comparisonHasMore
        ? "No other completed analyses on this page."
        : "No other completed analyses of this experiment yet. Run it again to compare results.";
      $("comparison-previous").hidden = offset === 0;
      $("comparison-next").hidden = !comparisonHasMore;
      picker.disabled = comparisonItems.length === 0;
      selection.comparisonLoaded = true;
    } catch (error) {
      if (active()) $("comparison-status").textContent = error.message;
    } finally {
      if (active()) {
        $("comparison-previous").disabled = comparisonOffset === 0;
        $("comparison-next").disabled = !comparisonHasMore;
      }
    }
  }

  function compare() {
    const selection = getSelection();
    const other = comparisonItems.find((item) => item.analysis_id === $("comparison-target").value);
    if (!selection?.metadata || !other || other.snapshot_id !== selection.metadata.snapshot_id
      || selection.metadata.job.state !== "succeeded") return;
    const path = `/api/v1/analyses/${selection.id}/comparison`;
    const query = `other=${encodeURIComponent(other.analysis_id)}`;
    $("comparison-preview").src = `${path}/preview?${query}`;
    $("comparison-preview").hidden = false;
    $("report-preview").hidden = true;
    $("download-comparison").href = `${path}?${query}`;
    $("download-comparison").download = `regimex-comparison-${selection.id}-${other.analysis_id}.html`;
    $("comparison-actions").hidden = false;
  }
  $("comparison-controls").addEventListener("toggle", () => {
    if ($("comparison-controls").open && !getSelection()?.comparisonLoaded) return comparisonPage();
  });
  $("comparison-target").addEventListener("change", () => {
    $("compare").disabled = !comparisonItems.some((item) => item.analysis_id === $("comparison-target").value);
  });
  $("comparison-previous").addEventListener("click", () =>
    comparisonPage(Math.max(0, comparisonOffset - COMPARISON_PAGE_SIZE)));
  $("comparison-next").addEventListener("click", () =>
    comparisonPage(comparisonOffset + COMPARISON_PAGE_SIZE));
  $("comparison-refresh").addEventListener("click", () => comparisonPage());
  $("compare").addEventListener("click", compare);
  $("close-comparison").addEventListener("click", () => {
    clearComparison();
    $("report-preview").hidden = !getSelection()?.metadata?.report_available;
  });
  return {reset};
}
