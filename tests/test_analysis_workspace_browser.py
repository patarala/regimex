"""Open market analyses from experiment history with keyboard or touch-sized layouts."""

from typing import Any
from uuid import UUID

import pytest
from test_offline_report_browser import browser as browser
from test_training_workspace_browser import ORIGIN, PREVIOUS, TrainingAPI


@pytest.mark.parametrize("width", [320, 1280])
def test_set_up_saved_experiment_and_open_its_result(browser: Any, width: int) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    snapshot_id, analysis_id = (str(UUID(int=value)) for value in (11, 12))
    created = "2026-01-01T12:00:00Z"
    experiment = {
        "snapshot_id": snapshot_id,
        "run_id": "morning",
        "display_name": "Morning market",
        "name_revision": 1,
        "instrument_ids": ["ETH/USD"],
        "step_count": 12,
        "created_at": created,
    }
    analysis = {
        **experiment,
        "analysis_id": analysis_id,
        "job": {
            "created_at": created,
            "state": "succeeded",
            "model": {"model_version_id": PREVIOUS},
        },
        "result_available": True,
        "report_available": False,
    }
    posts: list[dict[str, str]] = []

    def history(route: Any) -> None:
        count = len(posts)
        route.fulfill(
            json={
                "items": [
                    {
                        **experiment,
                        "analysis_count": count,
                        "successful_analysis_count": count,
                        "matching_analysis_count": count,
                        "latest_analysis": analysis if count else None,
                    }
                ]
            }
        )

    def start(route: Any) -> None:
        posts.append(route.request.post_data_json)
        assert route.request.headers["x-csrf-token"] == "test"
        assert route.request.headers["idempotency-key"]
        route.fulfill(status=202, json={"analysis_id": analysis_id})

    page = browser.new_page(viewport={"width": width, "height": 720})
    page.route(ORIGIN + "/**", TrainingAPI(role="user").dispatch)
    page.route("**/api/v1/experiments?*", history)
    page.route("**/api/v1/analyses", start)
    page.route(f"**/api/v1/analyses/{analysis_id}", lambda route: route.fulfill(json=analysis))
    try:
        page.goto(ORIGIN)
        setup = page.get_by_role("button", name="Set up analysis of Morning market")
        setup.focus()
        setup.press("Enter")
        chosen = page.locator("#file-name")
        expect(chosen).to_be_focused()
        expect(chosen).to_be_in_viewport()
        expect(page.locator("#file-summary")).to_have_text(
            "Saved experiment · morning · 12 steps · ETH/USD"
        )
        assert not posts
        page.get_by_role("button", name="Analyze", exact=True).click()
        title = page.get_by_role("heading", name="ETH/USD")
        expect(title).to_be_focused()
        expect(title).to_be_in_viewport()
        expect(page).to_have_url(f"{ORIGIN}/?analysis={analysis_id}")
        assert posts == [{"snapshot_id": snapshot_id, "model_version_id": PREVIOUS}]
        latest = page.get_by_role("button", name="Open latest analysis of Morning market")
        expect(latest).to_be_visible()
        page.go_back()
        expect(page.locator("#analysis-panel")).to_be_hidden()
        latest.focus()
        latest.press("Enter")
        expect(title).to_be_focused()
        expect(title).to_be_in_viewport()
    finally:
        page.close()


def test_status_resumes_after_leaving_analyses_during_a_temporary_error(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    analysis_id = str(UUID(int=41))
    path = f"/api/v1/analyses/{analysis_id}"
    attempts: list[str] = []
    state = "running"

    def status(route: Any) -> None:
        attempts.append(route.request.method)
        if len(attempts) == 1:
            route.fulfill(status=503, json={"error": {"message": "Service restarting."}})
            return
        route.fulfill(
            json={
                "analysis_id": analysis_id,
                "instrument_ids": ["BTC/USD"],
                "step_count": 12,
                "job": {
                    "state": state,
                    "stage": "predicting",
                    "operation_id": analysis_id,
                    "model": {"model_version_id": PREVIOUS},
                },
                "result_available": state == "succeeded",
                "report_available": False,
            }
        )

    page = browser.new_page()
    page.route(ORIGIN + "/**", TrainingAPI().dispatch)
    page.route(ORIGIN + path, status)
    try:
        page.clock.install()
        page.goto(f"{ORIGIN}/?analysis={analysis_id}")
        expect(page.locator("#job-guidance")).to_contain_text("Retrying automatically")
        page.get_by_role("link", name="Training", exact=True).click()
        expect(page.get_by_role("heading", name="Model training")).to_be_visible()
        reads = len(attempts)
        page.clock.fast_forward(60_000)
        assert len(attempts) == reads
        page.go_back()
        expect(page.locator("#job-state")).to_have_text("Running")
        expect(page.locator("#job-guidance")).to_be_hidden()
        state = "succeeded"
        page.clock.fast_forward(1500)
        expect(page.get_by_role("button", name="Download JSON")).to_be_visible()
        assert set(attempts) == {"GET"}
    finally:
        page.close()
