"""Exercise the saved HTML controls when Playwright and Chromium are installed."""

import importlib
import json
import math
import os
import shutil
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from support.report_fixtures import prediction_for
from test_analysis_comparison import _export, _render, _with_intervals

from regimex.directions import Direction
from regimex.exports import ExportPayloadV2, ExportV2, seal_export


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
    required = os.environ.get("REGIMEX_REQUIRE_BROWSER_TESTS") == "1"
    playwright = (
        importlib.import_module("playwright.sync_api")
        if required
        else pytest.importorskip("playwright.sync_api")
    )
    with playwright.sync_playwright() as driver:
        executable = os.environ.get("REGIMEX_TEST_CHROMIUM")
        if not required and executable is None:
            executable = shutil.which("chromium") or driver.chromium.executable_path
        if not required and executable is not None and not Path(executable).is_file():
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


def test_two_sandboxed_previews_share_one_plotting_runtime(browser: Any) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import HTMLResponse
    from test_analysis_report import digest, report_inputs
    from test_simulator_gateway_integration import tcp_server

    from regimex.api import _html_report
    from regimex.application.report import render_analysis_report
    from regimex.web import install_web_routes
    from regimex.web.report_runtime import plot_runtime, preview_html

    export, result = report_inputs()
    saved = render_analysis_report(export, result, result_sha256=digest(result)).html
    preview = preview_html(saved)
    runtime = plot_runtime()
    assert runtime is not None and len(saved) - len(preview) > 0.99 * len(runtime.script)
    app = FastAPI()
    install_web_routes(app)
    transfers: list[str] = []

    @app.middleware("http")
    async def count_assets(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path.startswith("/assets/plotly/"):
            transfers.append(request.url.path)
        return await call_next(request)

    @app.get("/preview/{analysis}")
    def serve_preview() -> Response:
        return _html_report(preview)

    @app.get("/history")
    def history() -> HTMLResponse:
        return HTMLResponse('<iframe sandbox="allow-scripts allow-downloads"></iframe>')

    with tcp_server(lambda origin: app) as server:
        page = browser.new_page()
        try:
            page.goto(server.origin + "/history")
            for report in ("first", "second"):
                url = server.origin + "/preview/" + report
                with page.expect_event(
                    "framenavigated", predicate=lambda frame, expected=url: frame.url == expected
                ):
                    page.locator("iframe").evaluate("(frame, url) => frame.src = url", url)
                frame = page.frame(url=url)
                assert frame is not None
                frame.locator("#probability .main-svg").first.wait_for()
                frame.get_by_role("button", name="Zoom to B/Q steps [2, 4)", exact=True).click()
                frame.wait_for_function(
                    """() => {
                        const node = document.getElementById("price");
                        return node.layout.xaxis.range[0] === 101
                            && node.layout.xaxis.range[1] === 102
                            && node.layout.yaxis.title.text.endsWith(' / B');
                    }"""
                )
                assert frame.evaluate("window.origin") == "null"
                frame.evaluate(
                    """() => new Promise(resolve => {
                      window.addEventListener('message', () => resolve(), {once: true});
                      window.postMessage({type: 'regimex-theme', theme: 'dark'}, '*');
                    })"""
                )
                assert frame.locator("html").get_attribute("data-theme") is None
                page.locator("iframe").evaluate(
                    """iframe => iframe.contentWindow.postMessage(
                      {type: 'regimex-theme', theme: 'dark'}, '*'
                    )"""
                )
                frame.wait_for_function(
                    """() => document.documentElement.dataset.theme === 'dark' &&
                        document.getElementById('price').layout.paper_bgcolor === '#1a1a1c'"""
                )
                assert frame.locator("#price").evaluate("plot => plot.layout.xaxis.range") == [
                    101,
                    102,
                ]
            assert transfers == [f"/assets/plotly/{runtime.digest}.js"]
        finally:
            page.close()


def test_model_comparison_tables_follow_system_theme_offline(browser: Any, tmp_path: Path) -> None:
    from test_comparison import evaluation

    from regimex.comparison import render_model_comparison

    target = tmp_path / "models"
    render_model_comparison((evaluation("baseline"), evaluation("tcn")), target)
    page = browser.new_page(offline=True, color_scheme="dark")
    try:
        page.goto((target / "index.html").as_uri())
        page.locator("tbody").first.wait_for()
        surface = page.locator("section").first
        dark = surface.evaluate("el => getComputedStyle(el).backgroundColor")
        page.emulate_media(color_scheme="light")
        page.wait_for_function(
            """dark => {
                const section = document.querySelector("section");
                return getComputedStyle(section).backgroundColor !== dark;
            }""",
            arg=dark,
        )
    finally:
        page.close()


@pytest.mark.parametrize("width", [1280, 390])
def test_offline_download_preserves_original_export(
    browser: Any, tmp_path: Path, width: int
) -> None:
    from test_analysis_report import digest, report_inputs

    from regimex.application.report import render_analysis_report, validate_analysis_result
    from regimex.storage import json_bytes

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
        validate_analysis_result(
            export, json_bytes(saved["result"]), result_sha256=saved["report"]["result_sha256"]
        )
        assert json_bytes(saved["export"]) == json_bytes(export.model_dump(mode="json"))
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
        selected = next(i for i in result.sequences[1].intervals if i.start <= 11 < i.stop)
        bounds = [100 + 0.5 * edge for edge in (selected.start, selected.stop)]
        assert page.locator("#timeline").evaluate("(node) => node.layout.xaxis.range") == bounds
        assert page.locator("#price").evaluate("(node) => node.layout.xaxis.range") == bounds
        row = page.get_by_role("button", name="Zoom to A/Q steps [20, 22)", exact=True)
        row.focus()
        row.press("Enter")
        page.wait_for_function(
            "document.getElementById('price').layout.yaxis.title.text.endsWith(' / A')"
        )
        for panel in ("price", "timeline", "probability", "volume", "activity", "coverage"):
            assert page.locator(f"#{panel}").evaluate("(node) => node.layout.xaxis.range") == [
                110,
                111,
            ]
        price = page.locator("#price")
        light_price = price.evaluate("plot => plot.data[0].line.color")
        page.emulate_media(color_scheme="dark")
        page.wait_for_function(
            "light => document.getElementById('price').data[0].line.color !== light",
            arg=light_price,
        )
        assert price.evaluate("plot => plot.layout.xaxis.range") == [110, 111]
        assert page.locator("#instrument").input_value() == "A/Q"
        page.emulate_media(color_scheme="light")
        page.wait_for_function(
            "light => document.getElementById('price').data[0].line.color === light",
            arg=light_price,
        )
        assert page.get_by_role("combobox").evaluate("(node) => document.activeElement === node")
        page.get_by_role("button", name="Reset zoom").click()
        page.wait_for_function("document.getElementById('coverage').layout.xaxis.range[0] === 100")
        assert page.locator("#plot-error").is_hidden()
    finally:
        page.close()


@pytest.mark.parametrize("width", [1280, 390])
def test_trend_pauses_on_the_price_chart(browser: Any, tmp_path: Path, width: int) -> None:
    from test_analysis_report import digest, pause_report_inputs

    from regimex.application.report import render_analysis_report

    export, result = pause_report_inputs()
    report = tmp_path / "analysis.html"
    report.write_bytes(render_analysis_report(export, result, result_sha256=digest(result)).html)
    touch = width < 600
    page = browser.new_page(viewport={"width": width, "height": 860}, has_touch=touch, offline=True)
    try:
        page.goto(report.as_uri())
        price = page.locator("#price")
        price.locator(".main-svg").first.wait_for()
        toggle = page.get_by_role("checkbox", name="Pauses within trends")
        assert not toggle.is_checked()
        assert price.evaluate("p => p.layout.shapes.length") == 0
        if touch:
            toggle.tap()
        else:
            toggle.focus()
            toggle.press("Space")
        band = price.locator(".shapelayer path")
        band.wait_for()
        assert price.evaluate("p => [p.layout.shapes[0].x0, p.layout.shapes[0].x1]") == [140, 156]
        page.evaluate("Plotly.Fx.hover('price', [{curveNumber: 0, pointNumber: 90}])")
        assert "32 steps · price change 0%" in price.locator(".hoverlayer").text_content()
        page.locator("#pause-count").click()
        zoom = page.get_by_role("button", name="Zoom to A/Q pause at steps [80, 112)")
        if touch:
            zoom.tap()
        else:
            zoom.press("Enter")
        for panel in ("price", "timeline", "probability"):
            assert page.locator(f"#{panel}").evaluate("p => p.layout.xaxis.range") == [140, 156]
        page.get_by_label("Instrument").select_option("B/Q")
        page.locator("#pause-empty").wait_for()
        band.wait_for(state="detached")
        assert page.locator("#pause-list button").count() == 0
        toggle.uncheck()
        assert page.locator("#pause-empty").is_hidden()
        assert page.locator("#plot-error").is_hidden()
    finally:
        page.close()


@pytest.mark.parametrize("width", [1280, 390])
def test_high_resolution_report_preserves_each_bin(
    browser: Any, tmp_path: Path, width: int
) -> None:
    from test_analysis_report import digest, report_inputs

    from regimex.application.report import render_analysis_report

    export, result = report_inputs(steps=100)
    grid = export.time_grid.model_copy(
        update={
            "origin": 1.7e18,
            "bin_width": 10.0,
            "completed_horizon": 1.700000000000001e18,
            "unit": "nanoseconds",
        }
    )
    payload = export.model_dump(exclude={"content_sha256"})
    export = seal_export(ExportPayloadV2(**{**payload, "time_grid": grid}))
    result = result.model_copy(update={"time_grid": grid, "snapshot_sha256": export.content_sha256})
    path = tmp_path / "high-resolution.html"
    path.write_bytes(render_analysis_report(export, result, result_sha256=digest(result)).html)
    page = browser.new_page(viewport={"width": width, "height": 860}, offline=True)
    try:
        page.goto(path.as_uri())
        page.locator("#price .main-svg").first.wait_for()
        price = page.locator("#price")
        assert price.evaluate("(node) => node.data[0].x") == [i + 0.5 for i in range(100)]
        assert price.evaluate("(node) => node.layout.xaxis.title.text") == "Bins from start"
        assert page.locator("#timeline").evaluate(
            "(node) => node.data.every(t => t.x.every(w => w > 0))"
        )
        page.evaluate("() => Plotly.relayout('price', {'xaxis.range': [71, 76]})")
        page.wait_for_function(
            "document.getElementById('probability').layout.xaxis.range[0] === 71"
        )
        page.get_by_role("button", name="Reset zoom").click()
        page.wait_for_function("document.getElementById('price').layout.xaxis.range[1] === 100")
    finally:
        page.close()


@pytest.mark.parametrize("width", [1280, 390])
def test_blind_review_shares_observations_under_an_opaque_id(
    browser: Any, tmp_path: Path, width: int
) -> None:
    from regimex.market_export import sample_snapshot
    from regimex.visualization import review_reference

    export = sample_snapshot(seed=1729, steps=64, instruments=2)
    sensitive_id = export.source.run_id
    output = tmp_path / "blind"
    review_reference(export, output)
    for artifact in output.iterdir():
        assert sensitive_id.encode() not in artifact.read_bytes()
    draft = json.loads((output / "annotation-draft.json").read_bytes())
    assert draft["experiment_id"] == export.content_sha256
    page = browser.new_page(viewport={"width": width, "height": 860}, offline=True)
    try:
        page.goto((output / "index.html").as_uri())
        page.locator("#explorer .main-svg").first.wait_for()
        assert page.get_by_label("Pauses within trends").is_hidden()
        page.get_by_label("Instrument", exact=True).select_option(index=1)
        assert export.content_sha256 in page.locator("#provenance").inner_text()
        page.get_by_role("link", name="Observations", exact=True).click()
        observations = json.loads(page.locator("body").inner_text())
        assert observations["experiment_id"] == draft["experiment_id"]
        assert observations["observations"] == export.model_dump(mode="json")["observations"]
    finally:
        page.close()


def test_single_bin_probabilities_and_submicrosecond_grid(browser: Any, tmp_path: Path) -> None:
    from test_analysis_report import digest
    from test_inspection import source

    from regimex.application.report import render_analysis_report
    from regimex.visualization import render_snapshot_report

    raw = _export(steps=1).model_dump(mode="json", exclude={"content_sha256"})
    raw["time_grid"].update(origin=0.0, bin_width=1e-9, completed_horizon=1e-9)
    raw["observations"][1]["trade_count"] = None
    export = seal_export(ExportPayloadV2.model_validate_json(json.dumps(raw)))
    result = prediction_for(export)
    single = tmp_path / "analysis.html"
    single.write_bytes(render_analysis_report(export, result, result_sha256=digest(result)).html)
    folder = tmp_path / "cli"
    render_snapshot_report(
        (source(0).model_copy(update={"export": export}),), folder, results=(result,)
    )
    page = browser.new_page(offline=True)
    try:
        page.goto(single.as_uri())
        points = page.locator("#probability .scatterlayer .point")
        points.first.wait_for(state="visible")
        assert points.count() == 3
        page.goto((folder / "index.html").as_uri())
        assert (
            page.locator("#time-guide").text_content()
            == "Step width: 1e-9 model_second · history [0, 1e-9)"
        )
        zoom = page.locator("#intervals button").first
        zoom.click()
        summary = page.locator("#focus-summary")
        assert "1 step ·" in summary.inner_text()
        assert "100% of steps with known counts had trades" in summary.inner_text()
        page.get_by_label("Instrument", exact=True).select_option(index=1)
        zoom.click()
        assert "Trade counts unavailable" in summary.inner_text()
    finally:
        page.close()


def test_comparison_retains_small_price_differences(browser: Any, tmp_path: Path) -> None:
    raw = _export(steps=4).model_dump(mode="json", exclude={"content_sha256"})
    for row in raw["observations"]:
        row["last_price"] = (10_000_001 + row["step"]) / 100
    export = seal_export(ExportPayloadV2.model_validate_json(json.dumps(raw)))
    first = prediction_for(export)
    second = prediction_for(export, direction=Direction.DECLINE)
    report = tmp_path / "comparison.html"
    report.write_bytes(_render(export, first, second))
    page = browser.new_page(offline=True)
    try:
        page.goto(report.as_uri())
        ticks = page.locator("#chart .price-tick")
        assert len(set(ticks.all_text_contents())) == 5
        assert ticks.evaluate_all("nodes => nodes.every(node => node.getBBox().x >= 0)")
        assert "Price 100000.01" in page.locator("#inspection").inner_text()
        page.locator("#inspect-step").fill("3")
        assert "Price 100000.04" in page.locator("#inspection").inner_text()
    finally:
        page.close()


def test_cli_pause_selection_keeps_zoom_on_instrument_change(browser: Any, tmp_path: Path) -> None:
    from test_analysis_report import pause_report_inputs
    from test_inspection import source

    from regimex.visualization import render_snapshot_report

    export, result = pause_report_inputs()
    output = tmp_path / "report"
    render_snapshot_report(
        (source(0).model_copy(update={"export": export}),), output, results=(result,)
    )
    page = browser.new_page(offline=True)
    try:
        page.goto((output / "index.html").as_uri())
        page.locator("#explorer .main-svg").first.wait_for()
        toggle = page.get_by_label("Pauses within trends")
        assert not toggle.is_checked()
        toggle.check()
        page.locator("#pause-count").click()
        page.locator("#pause-list button").click()
        axis_range = page.locator("#explorer").evaluate("plot => plot.layout.xaxis.range")
        assert abs(axis_range[0] - 79.5) <= 1 and abs(axis_range[1] - 111.5) <= 1
        page.get_by_label("Instrument", exact=True).select_option(label="B/Q")
        page.locator("#pause-empty").wait_for()
        assert page.locator("#explorer").evaluate("plot => plot.layout.xaxis.range") == axis_range
        assert page.locator("#pause-details").is_hidden()
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
        page.evaluate("() => Plotly.relayout('explorer', {'yaxis.range': [10, 30]})")
        page.emulate_media(color_scheme="dark")
        page.wait_for_function(
            "document.getElementById('explorer').layout.paper_bgcolor === '#1a1a1c'"
        )
        assert page.locator("#explorer").evaluate("plot => plot.layout.xaxis.range") == [1, 5]
        assert page.locator("#explorer").evaluate("plot => plot.layout.yaxis.range") == [10, 30]
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
def test_cli_report_draws_isolated_observations_without_connecting_gaps(
    browser: Any, tmp_path: Path, width: int
) -> None:
    from test_inspection import predictions, source

    from regimex.visualization import render_snapshot_report

    item = source(2, active=True)
    payload = item.export.model_dump(mode="json", exclude={"content_sha256"})
    payload["capabilities"]["price_policy"] = "execution_in_bin_only"
    payload["capabilities"]["historical_quotes"].update(
        mode="retained_snapshots", coverage="partial", provenance="Recorded quote at step 3."
    )
    for row in payload["observations"]:
        if row["step"] in (0, 2, 3, 5, 7):
            row.update(last_price=1e15 + row["step"] / 8)
        else:
            row.update(last_price=None, price_state="no_execution", trade_count=0, trade_volume=0.0)
        if row["step"] == 3:
            row.update(best_bid=104.0, bid_state="observed")
    item = item.model_copy(
        update={"export": seal_export(ExportPayloadV2.model_validate_json(json.dumps(payload)))}
    )
    quote_only = source(3, missing=True)
    quotes = quote_only.export.model_dump(mode="json", exclude={"content_sha256"})
    quotes["capabilities"]["historical_quotes"] = payload["capabilities"]["historical_quotes"]
    quotes["observations"][3].update(best_bid=104.0, bid_state="observed")
    quote_only = quote_only.model_copy(
        update={"export": seal_export(ExportPayloadV2.model_validate_json(json.dumps(quotes)))}
    )
    sources = (item, quote_only)
    output = tmp_path / "report"
    render_snapshot_report(sources, output, results=predictions(sources))
    page = browser.new_page(viewport={"width": width, "height": 860}, offline=True)
    try:
        page.goto((output / "index.html").as_uri())
        price = page.locator("#explorer .scatterlayer").first.locator(".trace").first
        price.wait_for(state="attached")
        assert page.locator(".chart-scroll").evaluate(
            "(node) => node.scrollWidth <= node.clientWidth + 1"
        )
        markers = price.locator(".point").evaluate_all(
            "(nodes) => nodes.filter(node => node.getBBox().width > 0).map(node => node.__data__.x)"
        )
        assert markers == [0, 5, 7]
        returns = page.locator("#explorer").evaluate(
            "plot => plot.data.find(trace => trace.name === 'Adjacent log return (%)').y"
        )
        expected = 100 * math.log1p(0.125 / (1e15 + 0.25))
        assert returns == [None] * 3 + [pytest.approx(expected, abs=0)] + [None] * 4
        assert (
            price.locator(".js-line").evaluate_all(
                "(paths) => paths.filter(path => path.getTotalLength() > 0).length"
            )
            == 1
        )

        page.get_by_label("Experiment", exact=True).select_option("snapshot-3")
        bid_point = page.locator("#explorer .scatterlayer").first.locator(".point")
        bid_point.wait_for()
        assert bid_point.evaluate("(node) => node.getBBox().width > 0")
        empty_panels = page.locator("#explorer").get_by_text("No observed values", exact=True)
        initially_empty = empty_panels.count()
        toggle = page.get_by_label("Best quotes")
        toggle.uncheck()
        bid_point.wait_for(state="hidden")
        assert empty_panels.count() == initially_empty + 1
        toggle.check()
        bid_point.wait_for()
        assert empty_panels.count() == initially_empty
    finally:
        page.close()


def test_report_switches_between_sources_with_the_same_snapshot_id(
    browser: Any, tmp_path: Path
) -> None:
    from test_inspection import predictions, source

    from regimex.visualization import render_snapshot_report

    first, second = source(0), source(1)
    payload = second.export.model_dump(mode="json", exclude={"content_sha256"})
    payload["source"]["snapshot_id"] = first.export.source.snapshot_id
    second = second.model_copy(
        update={
            "source_namespace": "second-installation",
            "export": seal_export(ExportPayloadV2.model_validate_json(json.dumps(payload))),
        }
    )
    output = tmp_path / "report"
    sources = (first, second)
    render_snapshot_report(sources, output, results=predictions(sources))
    page = browser.new_page(viewport={"width": 390, "height": 860}, offline=True)
    try:
        page.goto((output / "index.html").as_uri())
        selector = page.get_by_label("Experiment", exact=True)
        for item in (second, first):
            selector.select_option(item.export.content_sha256)
            page.locator("#provenance").get_by_text(
                item.export.content_sha256, exact=True
            ).wait_for()
            assert item.source_namespace in selector.locator("option:checked").inner_text()
    finally:
        page.close()


@pytest.mark.parametrize("width", [1280, 390])
def test_comparison_keeps_nanosecond_bins_distinct_when_zoomed(
    browser: Any, tmp_path: Path, width: int
) -> None:
    raw = _export(steps=100).model_dump(mode="json", exclude={"content_sha256"})
    raw["time_grid"].update(
        origin=1.7e18, bin_width=10.0, completed_horizon=1.700000000000001e18, unit="ns"
    )
    export = seal_export(ExportPayloadV2.model_validate_json(json.dumps(raw)))
    report = tmp_path / "comparison.html"
    report.write_bytes(
        _render(export, prediction_for(export), prediction_for(export, direction=Direction.DECLINE))
    )
    page = browser.new_page(viewport={"width": width, "height": 860}, offline=True)
    try:
        page.goto(report.as_uri())
        page.get_by_label("Start step").fill("10")
        page.get_by_label("Stop step").fill("14")
        page.get_by_role("button", name="Apply range").click()
        band = page.locator('#chart rect[y="232"]').first
        light_fill = band.evaluate("el => getComputedStyle(el).fill")
        page.emulate_media(color_scheme="dark")
        page.wait_for_function(
            """light => {
                const band = document.querySelector('#chart rect[y="232"]');
                return getComputedStyle(band).fill !== light;
            }""",
            arg=light_fill,
        )
        assert page.locator('#chart text[y="377"]').all_text_contents() == [
            "10",
            "11",
            "12",
            "13",
            "14",
        ]
        assert page.locator('#chart text[y="399"]').text_content() == "Bins from start"
        assert "Bin width: 10" in page.locator("#units").inner_text()
        for step in (10, 11, 12, 13):
            page.get_by_label("Inspect step").fill(str(step))
            assert (
                f"Bins from start [{step}, {step + 1})" in page.locator("#inspection").inner_text()
            )
        page.get_by_role("button", name="Reset zoom").click()
        assert page.get_by_label("Stop step").input_value() == "100"
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
    first = prediction_for(export).model_copy(update={"architecture_id": "persistence"})
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
        model_label = first.model.model_version_id + " · Segment MoE"
        assert page.locator("#provenance").get_by_text(model_label, exact=True).count() == 1

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
        page.get_by_label("Start step").fill("")
        page.get_by_role("button", name="Apply range").click()
        assert page.get_by_role("alert").is_visible()
        assert tick_offsets() == [40, 48.25, 56.5, 64.75, 73]
        # Inspect a point 31 time units into the zoomed 33-unit range.
        position = page.locator("#chart").evaluate("""(svg) => {
          const position = new DOMPoint(82 + (976 - 82) * 31 / 33, 135);
          const point = position.matrixTransform(svg.getScreenCTM());
          svg.dispatchEvent(new PointerEvent("pointermove", {clientX: point.x}));
          return document.getElementById("inspect-step").value;
        }""")
        assert position == "7"
        page.get_by_label("Start step").fill("7")
        page.get_by_label("Start step").press("Enter")
        assert tick_offsets() == [70, 70.75, 71.5, 72.25, 73]
        page.get_by_role("button", name="Reset zoom").click()
        assert page.locator("#range-start").input_value() == "0"
    finally:
        page.close()


def test_comparison_drag_ignores_other_pointers(browser: Any, tmp_path: Path) -> None:
    export = _export(steps=8)
    report = tmp_path / "comparison.html"
    report.write_bytes(
        _render(export, prediction_for(export), prediction_for(export, direction=Direction.DECLINE))
    )
    page = browser.new_page(offline=True)
    try:
        page.goto(report.as_uri())
        chart = page.locator("#chart")
        chart.scroll_into_view_if_needed()
        box = chart.bounding_box()
        y = box["y"] + box["height"] / 2
        page.mouse.move(box["x"] + box["width"] / 4, y)
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"] / 2, y)
        brush = chart.locator(".brush")
        width = brush.get_attribute("width")
        chart.dispatch_event("pointermove", {"pointerId": 99, "pointerType": "touch"})
        assert brush.get_attribute("width") == width
        chart.dispatch_event("pointercancel", {"pointerId": 99, "pointerType": "touch"})
        assert brush.is_visible()
        page.mouse.up()
        assert brush.count() == 0
        assert int(page.get_by_label("Start step").input_value()) > 0
        assert int(page.get_by_label("Stop step").input_value()) < 8
    finally:
        page.close()
