"""The browser can recover when deployment interrupts a workspace module request."""

from typing import Any

import pytest
from test_offline_report_browser import browser as browser
from test_training_workspace_browser import ORIGIN, TrainingAPI


def test_reload_recovers_a_deep_link_after_a_module_fails(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    page = browser.new_page()
    missing_module = f"{ORIGIN}/assets/ui.js"
    page.route(f"{ORIGIN}/**", TrainingAPI().dispatch)
    page.route(missing_module, lambda route: route.fulfill(status=404))
    try:
        url = f"{ORIGIN}/training?view=archived"
        page.goto(url)
        reload = page.get_by_role("link", name="Reload page")
        expect(reload).to_be_visible()
        assert reload.evaluate("(link) => link.href") == url

        page.unroute(missing_module)
        reload.press("Enter")
        expect(page.get_by_role("combobox", name="Run history view")).to_have_value("archived")
        expect(reload).to_be_hidden()
        assert page.url == url
    finally:
        page.close()
