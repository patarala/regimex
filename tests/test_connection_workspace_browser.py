"""Issue and hand off a simulator credential with the shipped Connections tab."""

from typing import Any
from urllib.parse import urlsplit
from uuid import UUID

import pytest
from test_offline_report_browser import browser as browser
from test_training_workspace_browser import ORIGIN, TrainingAPI


@pytest.mark.parametrize("width", [320, 1280])
def test_keyboard_key_handoff_and_backend_setup(browser: Any, width: int) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    connection_id, key_id = (str(UUID(int=i)) for i in (100, 101))
    connection = {
        "integration_id": connection_id,
        "display_name": "Market simulator",
        "source_namespace": "market",
        "owner_id": "desk",
        "enabled": True,
        "revision": 0,
    }
    key: dict[str, Any] = {
        "token_id": key_id,
        "name": "Overnight backend",
        "scopes": ["read", "upload", "analyze"],
        "expires_at": "2099-01-02T00:00:00Z",
        "created_at": "2026-01-02T00:00:00Z",
        "last_used_at": None,
        "revoked_at": None,
    }
    issued: list[dict[str, Any]] = []
    token = "rgx_test_once"
    api = "/api/v1/admin/integrations"
    detail = f"{api}/{connection_id}"

    def dispatch(route: Any) -> None:
        path = urlsplit(route.request.url).path
        if path == detail + "/keys" and route.request.method == "POST":
            assert route.request.headers["x-csrf-token"] == "test"
            issued.append(route.request.post_data_json)
            route.fulfill(status=201, json={"key": key, "token": token})
            return
        if path == f"{detail}/keys/{key_id}/revoke":
            assert route.request.method == "POST"
            assert route.request.headers["x-csrf-token"] == "test"
            key["revoked_at"] = "2026-03-01T00:00:00Z"
            route.fulfill(status=204)
            return
        responses = {
            api: {"items": [connection]},
            detail: connection,
            detail + "/keys": {"items": [key] if issued else []},
            detail + "/events": {"items": []},
            detail + "/storage": {"used_bytes": 0, "reserved_bytes": 0, "limit_bytes": 1024},
        }
        route.fulfill(json=responses[path])

    page = browser.new_page(viewport={"width": width, "height": 720})
    page.route(ORIGIN + "/**", TrainingAPI().dispatch)
    page.route(ORIGIN + api + "**", dispatch)
    page.add_init_script("Object.defineProperty(navigator, 'clipboard', {value: undefined});")
    try:
        page.goto(ORIGIN + "/connections")
        page.get_by_role("button", name="Market simulator").press("Enter")
        expect(page).to_have_url(f"{ORIGIN}/connections/{connection_id}")
        page.get_by_text("Backend setup", exact=True).press("Enter")
        expect(page.locator("#connection-verify-url")).to_have_text(ORIGIN + "/api/v1/connection")
        expect(page.locator("#connection-id")).to_have_text(connection_id)
        summary = page.get_by_text("Create service key", exact=True)
        summary.press("Enter")
        page.get_by_label("Key name").fill(str(key["name"]))
        page.get_by_role("button", name="Create key", exact=True).press("Enter")
        copy = page.get_by_role("button", name="Copy key")
        expect(copy).to_be_focused()
        secret = page.get_by_label("New service key · Overnight backend")
        expect(secret).to_have_attribute("type", "password")
        expect(page.locator("#connection-issued-id")).to_contain_text(key_id)
        copy.press("Enter")
        expect(page.locator("#connection-copy-status")).to_contain_text("Choose Show key")
        page.get_by_role("button", name="Show key").press("Enter")
        expect(secret).to_be_focused()
        assert (
            secret.evaluate("(el) => el.value.slice(el.selectionStart, el.selectionEnd)") == token
        )
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.get_by_role("button", name="Done", exact=True).press("Enter")
        expect(summary).to_be_focused()
        expect(page.locator("#connection-secret-value")).to_have_value("")
        expect(page.locator("#connection-keys")).to_contain_text(key_id)
        assert issued == [{"name": key["name"], "lifetime_days": 30, "scopes": key["scopes"]}]
        dialogs: list[str] = []

        def confirm(dialog: Any) -> None:
            dialogs.append(dialog.message)
            dialog.accept()

        page.once("dialog", confirm)
        page.get_by_role("button", name="Revoke").press("Enter")
        expect(page.locator("#connection-keys")).to_contain_text("Revoked")
        assert len(dialogs) == 1 and key_id in dialogs[0]
    finally:
        page.close()
