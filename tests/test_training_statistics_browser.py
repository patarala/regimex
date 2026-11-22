"""Exercise run accounting and independent result panels in the training workspace."""

from typing import Any

import pytest
from test_offline_report_browser import browser as browser
from test_training_workspace_browser import ORIGIN, RUN_ID, TrainingAPI


def test_timing_updates_preserve_selected_settings_and_freeze_at_completion(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.run["job"].update(state="queued", stage="recovery_pending", attempt=1, max_attempts=3)
    api.run["statistics"] = dict(elapsed_seconds=60, queued_seconds=59, running_seconds=1)
    page = browser.new_page(viewport={"width": 320, "height": 700})
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(f"{ORIGIN}/training/{RUN_ID}")
        expect(page.locator("#training-elapsed")).to_have_text("1 min 0 s")
        expect(page.locator("#training-stage")).to_have_text(
            "Waiting to restart after a worker interruption."
        )
        page.get_by_text("Run details and settings", exact=True).click()
        settings = page.locator("#training-settings")
        settings.evaluate(
            "(node) => getSelection().setBaseAndExtent(node.firstChild, 0, node.firstChild, 8)"
        )
        selected = page.evaluate("getSelection().toString()")
        api.run["job"].update(state="running", stage="loading", attempt=2)
        api.run["statistics"].update(elapsed_seconds=66, running_seconds=7)
        expect(page.locator("#training-running")).to_have_text("7 s", timeout=7000)
        expect(page.locator("#training-stage")).to_have_text(
            "Loading training data. Attempt 2 of 3."
        )
        assert selected and page.evaluate("getSelection().toString()") == selected

        api.run["job"]["state"] = "succeeded"
        api.run["statistics"].update(
            elapsed_seconds=68,
            running_seconds=9,
            fit_seconds=0.5,
            train_rows=72,
            validation_rows=24,
        )
        expect(page.locator("#training-elapsed")).to_have_text("1 min 8 s", timeout=7000)
        expect(page.locator("#training-queued")).to_have_text("59 s")
        facts = page.locator("#training-facts")
        for label, value in [
            ("Attempts", "2"),
            ("Model fit time", "<1 s"),
            ("Training rows", "72"),
            ("Validation rows", "24"),
        ]:
            expect(facts.get_by_text(label, exact=True).locator("+ dd")).to_have_text(value)
        assert page.locator("#training-timing").evaluate("n => n.scrollWidth <= n.clientWidth")
        page.clock.install()
        page.clock.fast_forward(60_000)
        expect(page.locator("#training-elapsed")).to_have_text("1 min 8 s")
    finally:
        page.close()


def test_historical_run_shows_elapsed_and_actions_when_metrics_are_unavailable(
    browser: Any,
) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.run["statistics"] = dict(elapsed_seconds=3605, queued_seconds=None, running_seconds=None)
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    page.route(
        f"**/training-runs/{RUN_ID}/metrics",
        lambda route: route.fulfill(status=404, json={"error": {"message": "Missing metrics."}}),
        times=1,
    )
    try:
        page.goto(f"{ORIGIN}/training/{RUN_ID}")
        expect(page.locator("#training-elapsed")).to_have_text("1 h 0 min")
        expect(page.locator("#training-queued")).to_be_hidden()
        expect(page.locator("#training-running")).to_be_hidden()
        expect(page.locator("#training-metric-error")).to_contain_text("Missing metrics.")
        expect(page.get_by_role("button", name="Evaluate model")).to_be_enabled()
        page.get_by_role("button", name="Refresh", exact=True).click()
        expect(page.get_by_role("img", name="Training loss by epoch")).to_be_visible()
        expect(page.locator("#training-metric-error")).to_be_hidden()
    finally:
        page.close()
