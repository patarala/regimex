"""Archive training history through its browser controls."""

from typing import Any
from uuid import UUID

import pytest
from test_offline_report_browser import browser as browser
from test_training_workspace_browser import ORIGIN, RUN_ID, TrainingAPI, completed_run


def test_archive_and_restore_a_published_run_without_losing_its_details(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.active_model = api.run["training"]["model_version_id"]
    api.runs.append(api.run)
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    detail_url = ORIGIN + "/#training/" + RUN_ID
    try:
        page.goto(detail_url)
        expect(page.get_by_text("New analyses use this model by default.")).to_be_visible()
        page.get_by_role("button", name="Archive run", exact=True).click()
        expect(page.get_by_text("No current runs.")).to_be_visible()
        expect(page.get_by_role("button", name="Restore run")).to_be_enabled()
        expect(page.locator("#training-created")).to_contain_text("Archived")
        expect(page.locator("#training-score")).to_contain_text("0.72")
        expect(page).to_have_url(detail_url)

        # The original bookmark works even when this run isn't in the displayed list.
        page.reload()
        expect(page.get_by_role("button", name="Restore run")).to_be_visible()
        view = page.get_by_role("combobox", name="Run history view")
        view.select_option("archived")
        expect(page.locator("#training-runs button[data-run-id]")).to_have_count(1)
        expect(page).to_have_url(detail_url + "?view=archived")
        page.go_back()
        expect(view).to_have_value("active")
        expect(page.get_by_text("No current runs.")).to_be_visible()
        page.go_forward()
        expect(view).to_have_value("archived")
        page.get_by_role("button", name="Restore run").click()
        expect(page.get_by_text("No archived runs.")).to_be_visible()
        expect(page.get_by_role("button", name="Archive run", exact=True)).to_be_enabled()
        expect(page.get_by_text("New analyses use this model by default.")).to_be_visible()
        assert api.archive_requests == [
            {"run_ids": [RUN_ID], "archived": True},
            {"run_ids": [RUN_ID], "archived": False},
        ]
    finally:
        page.close()


@pytest.mark.parametrize("use_filter", [True, False], ids=["filter", "deep-link"])
def test_navigation_during_archive_ignores_a_stale_detail(browser: Any, use_filter: bool) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.runs.append(api.run)
    old_row = api.run.copy()
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    detail_path = f"/api/v1/training-runs/{RUN_ID}"
    patches: list[Any] = []
    reads: list[Any] = []
    try:
        page.goto(f"{ORIGIN}/#training/{RUN_ID}")
        expect(page.get_by_role("button", name="Evaluate model")).to_be_enabled()
        page.route("**/training-runs/archive", lambda route: patches.append(route), times=1)
        with page.expect_request(lambda request: request.method == "PATCH"):
            page.get_by_role("button", name="Archive run", exact=True).click()
        view = page.get_by_role("combobox", name="Run history view")
        if use_filter:
            view.select_option("archived")
            expect(page.get_by_text("No archived runs.")).to_be_visible()
        page.route("**" + detail_path, lambda route: reads.append(route), times=1)
        with page.expect_request(lambda request: request.url.endswith(detail_path)):
            if use_filter:
                view.select_option("active")
            else:
                page.evaluate("hash => { location.hash = hash; }", f"#training/{RUN_ID}?view=all")
        expect(page.locator("#training-runs li")).to_have_count(1)
        api.dispatch(patches.pop())
        restore = page.get_by_role("button", name="Restore run")
        expect(restore).to_be_enabled()
        with page.expect_request_finished(lambda request: request.url.endswith(detail_path)):
            reads.pop().fulfill(json=old_row)
        expect(restore).to_be_enabled()
        expect(page.locator("#training-created")).to_contain_text("Archived")
    finally:
        page.close()


def test_bulk_archive_across_pages_and_restore_a_mixed_selection(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    for i in range(13):
        row = completed_run(str(UUID(int=100 + i)), "training")
        row["display_name"] = f"Trial {i}"
        row["job"]["state"] = "queued" if i == 1 else "failed"
        api.runs.append(row)
    page = browser.new_page(viewport={"width": 390, "height": 800})
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(ORIGIN + "/#training")
        expect(page.get_by_role("heading", name="Trial 0", exact=True)).to_be_visible()
        page.get_by_role("button", name="Run history", exact=True).click()
        view = page.get_by_role("combobox", name="Run history view")
        select = page.get_by_role("button", name="Select runs", exact=True)
        select.click()
        expect(page.get_by_role("checkbox", name="Select Trial 1", exact=True)).to_be_disabled()
        page.get_by_role("checkbox", name="Select Trial 0", exact=True).press("Space")
        page.get_by_role("button", name="Older").click()
        page.get_by_role("checkbox", name="Select Trial 12", exact=True).check()
        expect(page.get_by_text("2 selected", exact=True)).to_be_visible()
        page.get_by_role("button", name="Archive selected").click()
        expect(page.locator("#training-runs li")).to_have_count(11)
        expect(page.get_by_role("button", name="Newer")).to_be_hidden()
        view.select_option("all")
        select.click()
        expect(page.locator("#training-runs").get_by_text("Failed · Archived")).to_be_visible()
        page.get_by_role("checkbox", name="Select Trial 0", exact=True).check()
        page.get_by_role("checkbox", name="Select Trial 2", exact=True).check()
        expect(page.get_by_role("button", name="Archive 1 run")).to_be_enabled()
        page.get_by_role("button", name="Restore 1 run").click()
        expect(page.get_by_text("1 selected", exact=True)).to_be_visible()
        view.select_option("archived")
        expect(page.locator("#training-runs li")).to_have_count(1)
        expect(page.locator("#training-runs li")).to_contain_text("Trial 12")
        expect(select).to_be_visible()
        assert api.archive_requests == [
            {"run_ids": [api.runs[0]["run_id"], api.runs[12]["run_id"]], "archived": True},
            {"run_ids": [api.runs[0]["run_id"]], "archived": False},
        ]
    finally:
        page.close()
