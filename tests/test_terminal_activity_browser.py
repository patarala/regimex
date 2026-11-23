"""Read explicitly recorded terminal work through the shipped workspace."""

from typing import Any
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

import pytest
from test_offline_report_browser import browser as browser
from test_training_workspace_browser import ASSETS, ORIGIN, TrainingAPI


@pytest.mark.parametrize("role", ["admin", "user"])
def test_terminal_activity_navigation_polling_and_saved_text(browser: Any, role: str) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    rows: list[dict[str, Any]] = [
        {
            "activity_id": str(UUID(int=index + 1)),
            "title": f"Experiment {index + 1}",
            "started_at": "2026-11-23T08:00:00Z",
            "finished_at": "2026-11-23T09:00:00Z",
            "status": "succeeded",
            "exit_code": 0,
            "note": None,
            "result_text": None,
        }
        for index in range(21)
    ]
    selected = rows[-1]
    selected.update(status="running", finished_at=None, exit_code=None, note="November comparison")
    page = browser.new_page(viewport={"width": 320, "height": 760})
    page.clock.install()
    api = TrainingAPI(role=role)
    page.route(ORIGIN + "/**", api.dispatch)
    page.route(
        ORIGIN + "/activity**",
        lambda route: route.fulfill(path=ASSETS / "index.html", content_type="text/html"),
    )
    calls: list[str] = []

    def activity(route: Any) -> None:
        url = urlsplit(route.request.url)
        calls.append(url.path)
        assert route.request.method == "GET"
        if url.path.endswith("terminal-activities"):
            query = parse_qs(url.query)
            start, limit = int(query["offset"][0]), int(query["limit"][0])
            route.fulfill(
                json={"items": rows[start : start + limit], "offset": start, "limit": limit}
            )
        else:
            assert url.path.endswith(selected["activity_id"])
            route.fulfill(json=selected)

    page.route(ORIGIN + "/api/v1/terminal-activities**", activity)
    try:
        page.goto(ORIGIN)
        page.get_by_role("link", name="Activity", exact=True).click()
        history = page.get_by_role("list", name="Recorded commands")
        expect(history.get_by_role("link")).to_have_count(20)
        page.get_by_role("link", name="Older", exact=True).click()
        expect(history.get_by_role("link")).to_have_count(1)
        expect(page.get_by_role("link", name="Older", exact=True)).to_be_hidden()
        history.get_by_role("link").press("Enter")
        expect(page.get_by_role("heading", name="Experiment 21")).to_be_visible()
        expect(page.locator("#activity-facts")).to_contain_text("Running")
        expect(page.locator("#activity-note")).to_have_text("November comparison")
        expect(page.locator("#activity-result")).to_be_hidden()
        selected.update(
            status="failed",
            finished_at="2026-11-23T10:00:00Z",
            exit_code=5,
            result_text="first line\n<script>window.unwanted = true</script>",
        )
        page.clock.fast_forward(5000)
        expect(page.locator("#activity-facts")).to_contain_text("Exit code5")
        expect(page.locator("#activity-result")).to_have_text(selected["result_text"])
        assert page.locator("#activity-result script").count() == 0
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        before = len(calls)
        page.clock.fast_forward(6000)
        assert len(calls) == before
        page.reload()
        expect(page.locator("#activity-result")).to_have_text(selected["result_text"])
        page.get_by_role("link", name="All terminal activity").click()
        expect(page).to_have_url(ORIGIN + "/activity?offset=20")
        expect(history.get_by_role("link")).to_contain_text("Failed")
        page.get_by_role("link", name="Newer", exact=True).click()
        expect(history.get_by_role("link")).to_have_count(20)
        page.get_by_role("link", name="Analyses", exact=True).click()
        rows.clear()
        page.go_back()
        expect(page.locator("#activity-empty")).to_be_visible()
        expect(page.locator("#activity-list li")).to_have_count(0)
        rows.append(selected)
        page.evaluate("window.dispatchEvent(new Event('focus'))")
        page.clock.fast_forward(100)
        expect(history.get_by_role("link")).to_contain_text("Failed")
        if role == "user":
            expect(page.get_by_role("link", name="Training", exact=True)).to_be_hidden()
    finally:
        page.close()
