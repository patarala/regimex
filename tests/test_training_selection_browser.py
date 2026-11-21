"""Selecting runs keeps the history and its bulk controls in place."""

from typing import Any
from uuid import UUID

import pytest
from test_offline_report_browser import browser as browser
from test_training_workspace_browser import ORIGIN, TrainingAPI, completed_run


def toolbar_layout(page: Any) -> list[list[float]]:
    bounds: list[list[float]] = page.locator("#training-history").evaluate(
        """history => {
          const box = history.getBoundingClientRect();
          const list = history.querySelector("ul");
          const buttons = history.querySelectorAll("#training-selection button:not([hidden])");
          const relative = el => {
            const r = el.getBoundingClientRect();
            return [r.left - box.left, r.top - box.top, r.width, r.height];
          };
          return [
            relative(history.querySelector("#training-selection-count")),
            ...Array.from(buttons, relative),
            [relative(list.firstElementChild)[1] + list.scrollTop]
          ];
        }"""
    )
    return bounds


@pytest.mark.parametrize("width", [320, 821], ids=["phone", "narrow-sidebar"])
def test_bulk_actions_keep_their_place_from_empty_selection_to_limit(
    browser: Any, width: int
) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    for i in range(102):
        row = completed_run(str(UUID(int=100 + i)), "training")
        row["display_name"] = f"Trial {i}"
        row["job"]["state"] = "failed"
        row["archived_at"] = "2026-02-01T12:00:00Z" if i == 0 else None
        api.runs.append(row)
    page = browser.new_page(viewport={"width": width, "height": 800})
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(ORIGIN + "/training/new")
        if width == 320:
            page.get_by_role("button", name="Run history", exact=True).click()
        view = page.get_by_role("combobox", name="Run history view")
        select = page.get_by_role("button", name="Select runs", exact=True)
        archive = page.locator("#training-archive-selected")
        restore = page.locator("#training-restore-selected")
        for filter_value in ("active", "archived", "all"):
            view.select_option(filter_value)
            select.click()
            first = page.locator("#training-runs input").first
            for button, visible in (
                (archive, filter_value != "archived"),
                (restore, filter_value != "active"),
            ):
                if visible:
                    expect(button).to_be_visible()
                    expect(button).to_be_disabled()
                    button.evaluate("el => el.click()")
                else:
                    expect(button).to_be_hidden()
            empty = toolbar_layout(page)
            first.press("Space")
            expect(first).to_be_checked()
            expect(archive if filter_value == "active" else restore).to_be_enabled()
            if filter_value == "all":
                expect(archive).to_be_disabled()
            assert toolbar_layout(page) == empty
            first.uncheck()
            expect(page.get_by_text("0 selected", exact=True)).to_be_visible()
            assert toolbar_layout(page) == empty
            if filter_value != "all":
                page.get_by_role("button", name="Done", exact=True).click()

        first.check()
        page.locator("#training-runs input").nth(1).check()
        expect(archive).to_have_accessible_name("Archive selected (1 run)")
        expect(restore).to_have_accessible_name("Restore selected (1 run)")
        assert toolbar_layout(page) == empty

        # Check the same visible controls as a user, reacquiring each after the row rerenders.
        history = page.locator("#training-runs")
        for offset in range(0, 100, 12):
            if offset:
                page.get_by_role("button", name="Older", exact=True).click()
                expect(history.locator("strong").first).to_have_text(f"Trial {offset}")
            history.evaluate(
                """list => {
                  for (const item of list.children) {
                    const input = item.querySelector("input:enabled:not(:checked)");
                    if (input) input.click();
                  }
                }"""
            )
            assert toolbar_layout(page) == empty
        expect(page.get_by_text("100 of 100 selected", exact=True)).to_be_visible()
        expect(history.locator("input:not(:checked)")).to_have_count(2)
        expect(history.locator("input:not(:checked)").first).to_be_disabled()
        assert api.archive_requests == []
    finally:
        page.close()
