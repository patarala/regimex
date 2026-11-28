"""Archive training history through its browser controls."""

import re
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
    page = browser.new_page(locale="en-US", timezone_id="UTC")
    page.route(ORIGIN + "/**", api.dispatch)
    detail_url = ORIGIN + "/training/" + RUN_ID
    try:
        page.goto(detail_url)
        expect(page.get_by_text("New analyses use this model by default.")).to_be_visible()
        page.get_by_role("button", name="Archive run", exact=True).click()
        expect(page.get_by_text("No current runs.")).to_be_visible()
        expect(page.get_by_role("button", name="Restore run")).to_be_enabled()
        archived = page.locator("#training-archived")
        expect(archived).to_be_visible()
        expect(archived).to_have_text("Archived")
        page.get_by_text("Run details and settings", exact=True).click()
        expect(page.locator("#training-facts dt")).to_contain_text(["Archived"])
        expect(page.locator("#training-facts dd")).to_contain_text(["Feb 1, 2026"])
        expect(page.locator("#training-score")).to_contain_text("0.72")
        expect(page).to_have_url(detail_url)

        # The original bookmark works even when this run isn't in the displayed list.
        page.reload()
        expect(page.get_by_role("button", name="Restore run")).to_be_visible()
        view = page.get_by_role("combobox", name="Run history view")
        view.select_option("archived")
        row = page.locator("#training-runs button[data-run-id]")
        expect(row).to_have_count(1)
        expect(row.get_by_role("img", name="Archived")).to_have_count(0)
        expect(page).to_have_url(detail_url + "?view=archived")
        page.go_back()
        expect(view).to_have_value("active")
        expect(page.get_by_text("No current runs.")).to_be_visible()
        page.go_forward()
        expect(view).to_have_value("archived")
        page.get_by_role("button", name="Restore run").click()
        expect(page.get_by_text("No archived runs.")).to_be_visible()
        expect(page.get_by_role("button", name="Archive run", exact=True)).to_be_enabled()
        expect(archived).to_be_hidden()
        expect(page.get_by_text("New analyses use this model by default.")).to_be_visible()
        assert api.archive_requests == [
            {"run_ids": [RUN_ID], "archived": True},
            {"run_ids": [RUN_ID], "archived": False},
        ]
    finally:
        page.close()


def test_navigation_during_archive_ignores_a_stale_detail(browser: Any) -> None:
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
        page.goto(f"{ORIGIN}/training/{RUN_ID}?view=archived")
        expect(page.get_by_role("button", name="Evaluate model")).to_be_enabled()
        page.route("**/training-runs/archive", lambda route: patches.append(route), times=1)
        with page.expect_request(lambda request: request.method == "PATCH"):
            page.get_by_role("button", name="Archive run", exact=True).click()
        page.get_by_role("button", name="New run", exact=True).click()
        expect(page.get_by_role("heading", name="New training run")).to_be_visible()
        page.route("**" + detail_path, lambda route: reads.append(route), times=1)
        with page.expect_request(lambda request: request.url.endswith(detail_path)):
            page.go_back()
        expect(page.get_by_text("No archived runs.")).to_be_visible()
        api.dispatch(patches.pop())
        restore = page.get_by_role("button", name="Restore run")
        expect(restore).to_be_enabled()
        with page.expect_request_finished(lambda request: request.url.endswith(detail_path)):
            reads.pop().fulfill(json=old_row)
        expect(restore).to_be_enabled()
        expect(page.locator("#training-archived")).to_be_visible()
    finally:
        page.close()


def test_new_session_can_archive_while_previous_session_awaits_confirmation(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.runs.append(api.run)
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    page.route(
        "**/api/v1/session",
        lambda route: route.fulfill(
            json={
                "principal": {
                    "owner_id": "second" if route.request.method == "POST" else "first",
                    "role": "admin",
                },
                "csrf_token": "test",
            }
        ),
    )
    patches: list[Any] = []
    page.route("**/training-runs/archive", lambda route: patches.append(route))
    archive = page.get_by_role("button", name="Archive run", exact=True)
    try:
        page.goto(f"{ORIGIN}/training/{RUN_ID}")
        with page.expect_request(lambda request: request.method == "PATCH"):
            archive.click()
        page.get_by_role("button", name="Sign out").click()
        page.get_by_label("Access token").fill("test-token")
        page.get_by_role("button", name="Sign in").click()
        page.get_by_role("link", name="Training", exact=True).click()
        expect(archive).to_be_enabled()
        with page.expect_request(lambda request: request.method == "PATCH"):
            archive.click()
        with page.expect_response(lambda response: response.request.method == "PATCH"):
            patches[0].fulfill(status=204)
        expect(archive).to_be_disabled()
        api.dispatch(patches[1])
        expect(page.get_by_role("button", name="Restore run")).to_be_enabled()
    finally:
        page.close()


def test_filter_preserves_the_run_selected_repeat_and_epoch_table(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.runs.append(api.run)
    api.metrics["history"]["epochs"].extend(
        {"seed": 11, "epoch": epoch, "training_loss": 1 / epoch} for epoch in range(2, 26)
    )
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        url = f"{ORIGIN}/training/{RUN_ID}"
        page.goto(url)
        expect(page.get_by_role("button", name="Evaluate model")).to_be_enabled()
        seed = page.get_by_label("Training repeat")
        seed.select_option("11")
        page.locator("#training-run-details > summary").click()
        page.locator('summary[aria-label="Training loss values"]').click()
        page.get_by_role("button", name="Next values").click()
        table = page.get_by_role("table", name="Training loss by epoch")
        chart = page.get_by_role("img", name="Training loss by epoch").element_handle()
        view = page.get_by_role("combobox", name="Run history view")
        rows = page.locator("#training-runs li")
        before = len(api.pages)

        def expect_preserved(filter_name: str) -> None:
            expect(view).to_have_value(filter_name)
            expect(page).to_have_url(url + f"?view={filter_name}")
            expect(rows).to_have_count(0 if filter_name == "archived" else 1)
            expect(seed).to_have_value("11")
            expect(table.get_by_role("rowheader").first).to_have_text("21")
            expect(page.locator("#training-settings")).to_be_visible()
            assert chart.evaluate("svg => svg.isConnected")

        view.select_option("archived")
        expect(page.get_by_text("No archived runs.")).to_be_visible()
        expect_preserved("archived")
        view.select_option("all")
        expect_preserved("all")
        page.go_back()
        expect_preserved("archived")
        page.go_forward()
        expect_preserved("all")
        assert api.pages[before:] == ["training-runs"] * 4
    finally:
        page.close()


def test_filter_preserves_unfinished_run_json_on_a_phone(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.runs.append(api.run)
    page = browser.new_page(viewport={"width": 375, "height": 820})
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        url = ORIGIN + "/training/new"
        page.goto(url)
        expect(page.get_by_role("button", name="Start training")).to_be_enabled()
        page.get_by_role("button", name="Edit JSON").click()
        editor = page.get_by_label("Run configuration (JSON)")
        draft = '{"policy":{"epochs":'
        editor.fill(draft)
        page.get_by_role("button", name="Run history", exact=True).click()
        view = page.get_by_role("combobox", name="Run history view")
        before = len(api.pages)
        view.focus()
        view.select_option("archived")
        expect(page.get_by_text("No archived runs.")).to_be_visible()
        expect(page).to_have_url(url + "?view=archived")
        expect(editor).to_be_visible()
        expect(editor).to_have_value(draft)
        expect(view).to_be_focused()
        page.go_back()
        expect(page).to_have_url(url)
        expect(page.locator("#training-runs li")).to_have_count(1)
        expect(editor).to_be_visible()
        expect(editor).to_have_value(draft)
        assert api.pages[before:] == ["training-runs", "training-runs"]
    finally:
        page.close()


@pytest.mark.parametrize("stale_status", [200, 503])
def test_a_slower_filter_keeps_old_rows_until_the_latest_view_loads(
    browser: Any, stale_status: int
) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.run["display_name"] = "Current trial"
    api.run["job"].update(state="running", stage="training", attempt=1)
    api.progress_rows = [{"seed": 11, "epoch": 1, "training_loss": 0.9}]
    archived = completed_run(str(UUID(int=91)), "training")
    archived.update(display_name="Older trial", archived_at="2026-02-01T12:00:00Z")
    api.runs = [api.run, archived]
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    pending: list[Any] = []
    try:
        page.clock.install()
        page.goto(f"{ORIGIN}/training/{RUN_ID}")
        history = page.locator("#training-runs")
        current = history.get_by_text("Current trial", exact=True)
        expect(current).to_be_visible()
        chart = page.get_by_role("img", name="Training loss by epoch")
        expect(chart).to_be_visible()
        view = page.get_by_role("combobox", name="Run history view")
        page.route(
            "**/training-runs?archive=archived&*", lambda route: pending.append(route), times=1
        )
        with page.expect_request(lambda request: "archive=archived&" in request.url):
            view.select_option("archived")
        expect(history).to_have_attribute("aria-busy", "true")
        expect(current).to_be_visible()
        api.progress_rows.append({"seed": 11, "epoch": 2, "training_loss": 0.4})
        page.clock.fast_forward(3000)
        expect(chart.locator("desc")).to_contain_text("Epoch 2: 0.4")
        view.select_option("all")
        expect(history.get_by_text("Older trial", exact=True)).to_be_visible()
        expect(history).to_have_attribute("aria-busy", "false")
        with page.expect_request_finished(lambda request: "archive=archived&" in request.url):
            route = pending.pop()
            if stale_status == 200:
                api.dispatch(route)
            else:
                route.fulfill(status=503, json={"error": {"message": "Archive unavailable."}})
        expect(history.locator("li")).to_have_count(2)
        expect(view).to_have_value("all")
        expect(page.locator("#training-notice")).not_to_contain_text("Archive unavailable.")
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
        page.goto(ORIGIN + "/training")
        expect(page.get_by_role("heading", name="Trial 0", exact=True)).to_be_visible()
        page.get_by_role("button", name="Run history", exact=True).click()
        view = page.get_by_role("combobox", name="Run history view")
        select = page.get_by_role("button", name="Select runs", exact=True)
        select.click()
        pending = page.get_by_role("checkbox", name="Select Trial 1", exact=True)
        pending.check()
        expect(page.get_by_role("button", name="Archive selected")).to_be_disabled()
        pending.uncheck()
        page.get_by_role("checkbox", name="Select Trial 0", exact=True).press("Space")
        page.get_by_role("button", name="Older").click()
        page.get_by_role("checkbox", name="Select Trial 12", exact=True).check()
        expect(page.get_by_text("2 selected", exact=True)).to_be_visible()
        page.get_by_role("button", name="Archive selected").click()
        expect(page.locator("#training-runs li")).to_have_count(11)
        expect(page.get_by_role("button", name="Newer")).to_be_hidden()
        view.select_option("all")
        select.click()
        archived = page.locator(f'button[data-run-id="{api.runs[0]["run_id"]}"]')
        expect(archived.locator("[data-state]")).to_have_text("Failed")
        expect(archived).to_have_accessible_name(re.compile(r"\bArchived\b"))
        page.get_by_role("checkbox", name="Select Trial 0", exact=True).check()
        page.get_by_role("checkbox", name="Select Trial 2", exact=True).check()
        expect(page.get_by_role("button", name="Archive selected (1 run)")).to_be_enabled()
        page.get_by_role("button", name="Restore selected (1 run)").click()
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
