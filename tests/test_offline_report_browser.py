"""Exercise the saved HTML controls when Playwright and Chromium are installed."""

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from support.report_fixtures import prediction_for
from test_analysis_comparison import _export, _render, _with_intervals

from regimex.directions import Direction
from regimex.exports import ExportPayloadV2, ExportV2, seal_export


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as driver:
        executable = shutil.which("chromium") or driver.chromium.executable_path
        if not Path(executable).is_file():
            pytest.skip("Install Chromium to exercise offline reports")
        instance = driver.chromium.launch(executable_path=executable)
        try:
            yield instance
        finally:
            instance.close()


def click_visible_lane(page: Any, selector: str, lane: int) -> None:
    """Click a real bar at the center of the visible x range."""
    plot = page.locator(selector)
    position = plot.evaluate(
        """(node, lane) => {
      const trace = node.data.find(item =>
        item.meta === "instrument-lane" && item.y.includes(lane));
      const layout = node._fullLayout;
      const x = layout.xaxis;
      const y = layout["yaxis" + (trace.yaxis || "y").slice(1)];
      return {x: x._offset + x._length / 2, y: y._offset + y.l2p(lane)};
    }""",
        lane,
    )
    plot.click(position=position)


@pytest.mark.parametrize("width", [1280, 390])
def test_offline_download_preserves_original_export(
    browser: Any, tmp_path: Path, width: int
) -> None:
    from test_analysis_report import digest, report_inputs

    from regimex.application.report import render_analysis_report

    export, result = report_inputs(missing=True, interleaved=True)
    report = tmp_path / "analysis.html"
    report.write_bytes(render_analysis_report(export, result, result_sha256=digest(result)).html)
    page = browser.new_page(viewport={"width": width, "height": 860}, offline=True)
    try:
        page.goto(report.as_uri())
        with page.expect_download() as download:
            page.get_by_role("button", name="Download JSON").click()
        saved = json.loads(download.value.path().read_bytes())
        assert ExportV2.model_validate_json(json.dumps(saved["export"])) == export
        assert saved["result"] == result.model_dump(mode="json")
        page.locator("#coverage .main-svg").first.wait_for()
        page.evaluate("() => Plotly.relayout('price', {'xaxis.range': [103, 108]})")
        page.wait_for_function(
            "document.getElementById('probability').layout.xaxis.range[0] === 103"
        )
        assert page.locator("#timeline .ytick text").all_text_contents() == ["A/Q", "B/Q"]
        click_visible_lane(page, "#timeline", 1)
        page.wait_for_function("document.getElementById('instrument').value === 'B/Q'")
        page.wait_for_function(
            "document.getElementById('price').layout.yaxis.title.text.endsWith(' / B')"
        )
        assert page.locator("#timeline").evaluate("(node) => node.layout.xaxis.range") == [103, 108]
        assert page.locator("#price").evaluate("(node) => node.layout.xaxis.range") == [103, 108]
        page.get_by_role("combobox").select_option("A/Q")
        page.wait_for_function("document.getElementById('instrument').value === 'A/Q'")
        assert page.locator("#price").evaluate("(node) => node.layout.xaxis.range") == [103, 108]
        page.get_by_role("button", name="Reset zoom").click()
        page.wait_for_function("document.getElementById('coverage').layout.xaxis.range[0] === 100")
        assert page.locator("#plot-error").is_hidden()
    finally:
        page.close()


@pytest.mark.parametrize("width", [1280, 390])
def test_cli_report_switches_lanes_within_the_zoomed_experiment(
    browser: Any, tmp_path: Path, width: int
) -> None:
    from test_inspection import predictions, source

    from regimex.visualization import render_snapshot_report

    sources = (source(1, instrument_count=2, active=True), source(2, active=True))
    output = tmp_path / "report"
    render_snapshot_report(sources, output, results=predictions(sources))
    page = browser.new_page(viewport={"width": width, "height": 860}, offline=True)
    try:
        page.goto((output / "index.html").as_uri() + "#experiment=snapshot-1")
        page.locator("#explorer .barlayer path").first.wait_for()
        assert page.locator(".chart-scroll").evaluate(
            "(node) => node.scrollWidth <= node.clientWidth + 1"
        )
        page.evaluate("() => Plotly.relayout('explorer', {'xaxis.range': [1, 5]})")
        click_visible_lane(page, "#explorer", 1)
        page.wait_for_function(
            "document.getElementById('sample').selectedOptions[0].text === 'ETH/USD'"
        )
        assert "ETH" in page.locator("#sample-meta").inner_text()
        assert page.locator("#explorer").evaluate("(node) => node.layout.xaxis.range") == [1, 5]
        page.get_by_label("Instrument").select_option(label="BTC/USD")
        assert page.locator("#explorer").evaluate("(node) => node.layout.xaxis.range") == [1, 5]
        page.get_by_label("Experiment", exact=True).select_option("snapshot-2")
        page.wait_for_function("document.getElementById('timeline').hidden === false")
        assert page.locator("#explorer").evaluate("(node) => node.layout.xaxis.range") == [
            -0.5,
            7.5,
        ]
        assert page.locator("#explorer .barlayer path").count() == 0
        assert page.locator("#render-error").is_hidden()
    finally:
        page.close()


@pytest.mark.parametrize("width", [1280, 390])
def test_difference_zoom_and_partial_terminal_bin(browser: Any, tmp_path: Path, width: int) -> None:
    raw = _export(steps=8).model_dump(mode="json", exclude={"content_sha256"})
    raw["time_grid"].update(
        origin=1700000000.0,
        bin_width=10.0,
        completed_horizon=1700000073.0,
        terminal_bin="include_partial_bin",
    )
    export = seal_export(ExportPayloadV2.model_validate_json(json.dumps(raw)))
    first = prediction_for(export)
    second = _with_intervals(
        prediction_for(export, direction=Direction.DECLINE),
        export,
        ((0, 4, Direction.GROWTH), (4, 8, Direction.DECLINE)),
    )
    report = tmp_path / "comparison.html"
    report.write_bytes(_render(export, first, second))
    page = browser.new_page(viewport={"width": width, "height": 860}, offline=True)
    try:
        page.goto(report.as_uri())

        def tick_offsets() -> list[float]:
            ticks = page.locator('#chart text[y="377"]').all_text_contents()
            return [float(tick) - export.time_grid.origin for tick in ticks]

        assert tick_offsets() == [0, 18.25, 36.5, 54.75, 73]
        band = page.locator('#chart rect:has(title:text("direction difference"))')
        # Four changed bins occupy 33 of the 73 time units on the original grid.
        assert float(band.get_attribute("width")) == pytest.approx((976 - 82) * 33 / 73)
        zoom = page.get_by_role("button", name="Zoom to steps 4 through 7")
        zoom.focus()
        zoom.press("Enter")
        assert page.locator("#range-start").input_value() == "4"
        assert tick_offsets() == [40, 48.25, 56.5, 64.75, 73]
        assert page.locator("#chart-container").evaluate(
            "(node) => node === document.activeElement"
        )
        # Inspect a point 31 time units into the zoomed 33-unit range.
        position = page.locator("#chart").evaluate("""(svg) => {
          const position = new DOMPoint(82 + (976 - 82) * 31 / 33, 135);
          const point = position.matrixTransform(svg.getScreenCTM());
          svg.dispatchEvent(new PointerEvent("pointermove", {clientX: point.x}));
          return document.getElementById("inspect-step").value;
        }""")
        assert position == "7"
        page.get_by_label("Start step").fill("7")
        page.get_by_role("button", name="Apply range").click()
        assert tick_offsets() == [70, 70.75, 71.5, 72.25, 73]
        page.get_by_role("button", name="Reset zoom").click()
        assert page.locator("#range-start").input_value() == "0"
    finally:
        page.close()
