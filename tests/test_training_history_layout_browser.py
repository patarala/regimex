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
