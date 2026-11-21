"""Run history remains readable at the narrowest sidebar and phone layouts."""

from typing import Any

import pytest
from test_offline_report_browser import browser as browser
from test_training_workspace_browser import ORIGIN, RUN_ID, TrainingAPI, completed_run


@pytest.mark.parametrize("width", [821, 320])
def test_run_history_shows_full_labels_and_keeps_time_together(browser: Any, width: int) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    name = "Сравнение валют " + "W" * 84
    dataset = "market_" + "momentum" * 12
    api.runs = [api.run, completed_run("00000000-0000-0000-0000-000000000002", "training")]
    for run in api.runs:
        run["display_name"] = name
        run["training"]["dataset_id"] = dataset
        run["trainable_parameters"] = 120000
        run["job"]["created_at"] = "2026-09-30T23:59:00Z"

    page = browser.new_page(
        viewport={"width": width, "height": 900}, locale="en-US", timezone_id="UTC"
    )
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        # Older dates include the year even in the narrowest selection layout.
        page.clock.set_fixed_time("2027-01-01T13:00:00Z")
        page.goto(f"{ORIGIN}/#training/new")
        if width == 320:
            page.get_by_role("button", name="Run history", exact=True).click()
        row = page.locator(f'#training-runs button[data-run-id="{RUN_ID}"]')
        expect(row.locator("strong")).to_have_text(name)
        expect(row.locator("time")).to_contain_text("PM")
        expect(row.locator("time")).to_contain_text("2026")
        for selecting in (False, True):
            if selecting:
                page.get_by_role("button", name="Select runs", exact=True).click()
                page.locator(f'#training-runs input[data-select-run="{RUN_ID}"]').check()
            layout = row.evaluate(
                """button => {
                  const bounds = button.getBoundingClientRect();
                  return ['strong', 'time', '[data-state]'].map(selector => {
                    const element = button.querySelector(selector);
                    const range = document.createRange();
                    range.selectNodeContents(element);
                    const lines = [...range.getClientRects()].filter(rect => rect.width);
                    const box = element.getBoundingClientRect();
                    return {
                      lines: new Set(lines.map(rect => rect.top)).size,
                      fits: lines.every(rect =>
                        rect.left >= bounds.left && rect.right <= bounds.right &&
                        rect.top >= box.top - 1 && rect.bottom <= box.bottom + 1)
                    };
                  });
                }"""
            )
            assert all(item["fits"] for item in layout), layout
            title, timestamp, state = layout
            assert title["lines"] > 2
            assert timestamp["lines"] == state["lines"] == 1
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    finally:
        page.close()


@pytest.mark.parametrize("width", [1280, 375, 320])
def test_history_tooltip_keeps_parameter_counts_and_dataset_ids_readable(
    browser: Any, width: int
) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    canonical = "dataset-12345678abcdef0012345678abcdef00"
    long_id = "dataset-" + "WideIdentifier" * 8
    api.runs = [api.run, completed_run("00000000-0000-0000-0000-000000000002", "training")]
    for run, dataset in zip(api.runs, (canonical, long_id), strict=True):
        run.update(display_name=dataset[:18], trainable_parameters=135555)
        run["training"]["dataset_id"] = dataset
    page = browser.new_page(viewport={"width": width, "height": 800}, locale="en-US")
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(f"{ORIGIN}/#training/new")
        if width < 821:
            page.get_by_role("button", name="Run history", exact=True).click()
        for run in api.runs:
            dataset = run["training"]["dataset_id"]
            page.locator(f'[data-run-info="{run["run_id"]}"]').click()
            tooltip = page.get_by_role("tooltip")
            expect(tooltip.locator(".run-info-size")).to_have_text("135,555 parameters")
            expect(tooltip.locator(".run-info-dataset")).to_have_text(dataset)
            layout = tooltip.evaluate(
                """tip => {
                  function fragments(selector, end) {
                    const node = tip.querySelector(selector).firstChild;
                    const range = document.createRange();
                    range.setStart(node, 0);
                    range.setEnd(node, end ?? node.length);
                    return [...range.getClientRects()];
                  }
                  const id = fragments('.run-info-dataset');
                  const prefix = fragments('.run-info-dataset', 'dataset-'.length + 1);
                  const size = fragments('.run-info-size');
                  const bounds = tip.getBoundingClientRect();
                  return {
                    sizeLines: size.length,
                    idLines: id.length,
                    prefixTogether: prefix.length === 1,
                    inBounds: bounds.left >= 0 && bounds.right <= innerWidth &&
                      tip.scrollWidth <= tip.clientWidth &&
                      [...id, ...size].every(r => r.left >= bounds.left && r.right <= bounds.right)
                  };
                }"""
            )
            assert layout["inBounds"] and layout["sizeLines"] == 1, layout
            assert layout["prefixTogether"], layout
            if dataset == long_id:
                assert layout["idLines"] > 1, layout
            else:
                assert layout["idLines"] == 1, layout
            page.keyboard.press("Escape")
    finally:
        page.close()
