"""Workspace startup, navigation and session recovery in Chromium."""

from typing import Any

import pytest
from test_offline_report_browser import browser as browser
from test_training_workspace_browser import ORIGIN, TrainingAPI


@pytest.mark.parametrize(("width", "theme"), [(320, "dark"), (1280, "light")])
def test_workspace_navigation_identifies_the_current_page(
    browser: Any, width: int, theme: str
) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    page = browser.new_page(viewport={"width": width, "height": 800}, color_scheme=theme)
    page.route(ORIGIN + "/**", TrainingAPI().dispatch)
    page.route("**/api/v1/admin/integrations?*", lambda route: route.fulfill(json={"items": []}))
    owner = "desk-" + "a" * 123
    page.route(
        "**/api/v1/session",
        lambda route: route.fulfill(
            json={"principal": {"owner_id": owner, "role": "admin"}, "csrf_token": "test"}
        ),
    )
    nav = page.get_by_role("navigation", name="Workspace")

    def check_page(name: str) -> None:
        expect(page).to_have_title(f"{name} · Regimex")
        expect(nav.locator('[aria-current="page"]')).to_have_text(name)
        expect(page.get_by_role("heading", level=1, name=name, exact=True)).to_be_focused()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")

    try:
        page.goto(ORIGIN + "/training")
        check_page("Training")
        account = page.locator("#owner")
        expect(account).to_have_attribute("title", owner)
        if width == 320:
            assert account.evaluate("node => node.scrollWidth <= node.clientWidth")
        nav.get_by_text("Connections").press("Enter")
        check_page("Connections")
        nav.get_by_text("Analyses").press("Enter")
        check_page("Analyses")
        page.go_back()
        check_page("Connections")
        page.go_forward()
        check_page("Analyses")
        page.route("**/api/v1/experiments?*", lambda route: route.fulfill(status=401, json={}))
        page.get_by_role("region", name="Experiments").get_by_role("button", name="Refresh").click()
        expect(page.get_by_role("heading", level=1, name="Sign in")).to_be_focused()
        expect(page).to_have_title("Sign in · Regimex")
        expect(page.locator("#notice")).to_contain_text("Your session has expired.")
        expect(nav).to_be_hidden()
    finally:
        page.close()


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
