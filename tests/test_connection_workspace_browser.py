"""Issue and hand off a simulator credential with the shipped Connections tab."""

from typing import Any
from urllib.parse import urlsplit
from uuid import UUID

import pytest
from test_offline_report_browser import browser as browser
from test_training_workspace_browser import ORIGIN, TrainingAPI


def test_connection_source_id_preserves_unicode_at_the_api_limit(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    page = browser.new_page()
    page.route(ORIGIN + "/**", TrainingAPI().dispatch)
    page.route("**/api/v1/admin/integrations?*", lambda route: route.fulfill(json={"items": []}))
    try:
        page.goto(ORIGIN + "/connections")
        page.locator("#connection-create-details summary").click()
        form = page.locator("#connection-create")
        form.get_by_label("Name", exact=True).fill("Simulator")
        form.get_by_label("Integration account").fill("desk")
        source = form.get_by_label("Export source ID")
        identifier = "𠀋" * 128
        source.focus()
        page.keyboard.insert_text(identifier + "x")
        expect(source).to_have_value(identifier + "x")
        assert not source.evaluate("input => input.checkValidity()")
        source.press("Backspace")
        expect(source).to_have_value(identifier)
        assert source.evaluate("input => input.checkValidity()")
        page.route("**/api/v1/admin/integrations", lambda route: route.fulfill(status=409, json={}))
        with page.expect_request(ORIGIN + "/api/v1/admin/integrations") as sent:
            form.get_by_role("button", name="Add connection").click()
        assert sent.value.post_data_json["source_namespace"] == identifier
    finally:
        page.close()


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
    page.context.route(ORIGIN + "/**", TrainingAPI().dispatch)
    page.context.route(ORIGIN + api + "**", dispatch)
    page.add_init_script("Object.defineProperty(navigator, 'clipboard', {value: undefined});")
    try:
        page.goto(ORIGIN + "/connections")
        link = page.get_by_role("link", name="Market simulator")
        expect(link).to_have_attribute("href", f"/connections/{connection_id}")
        with page.context.expect_page() as opened:
            link.click(modifiers=["ControlOrMeta"])
        popup = opened.value
        expect(popup.get_by_role("heading", name="Market simulator")).to_be_visible()
        popup.close()
        expect(page).to_have_url(ORIGIN + "/connections")
        link.press("Enter")
        expect(page).to_have_url(f"{ORIGIN}/connections/{connection_id}")
        expect(link).to_have_attribute("aria-current", "page")
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


def test_concurrent_edits_keep_the_name_and_require_a_fresh_decision(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    connection_id = str(UUID(int=84))
    api = "/api/v1/admin/integrations"
    detail = f"{api}/{connection_id}"
    connection: dict[str, Any] = {
        "integration_id": connection_id,
        "display_name": "Simulator",
        "source_namespace": "market",
        "owner_id": "desk",
        "revision": 0,
        "enabled": True,
    }
    requests: list[dict[str, Any]] = []

    def dispatch(route: Any) -> None:
        path = urlsplit(route.request.url).path
        if route.request.method == "PATCH":
            assert path == detail
            assert route.request.headers["x-csrf-token"] == "test"
            fields = route.request.post_data_json
            requests.append(fields.copy())
            if fields.pop("expected_revision") != connection["revision"]:
                route.fulfill(status=409, json={"error": {"message": "Stale revision."}})
                return
            connection.update(fields)
            connection["revision"] += 1
        responses = {
            api: {"items": [connection]},
            detail: connection,
            detail + "/keys": {"items": []},
            detail + "/events": {"items": []},
            detail + "/storage": {"used_bytes": 0, "reserved_bytes": 0, "limit_bytes": 1024},
        }
        route.fulfill(json=responses[path])

    context = browser.new_context()
    context.route(ORIGIN + "/**", TrainingAPI().dispatch)
    context.route(ORIGIN + api + "**", dispatch)
    first, second = context.new_page(), context.new_page()
    try:
        for page in (first, second):
            page.goto(f"{ORIGIN}/connections/{connection_id}")
            expect(page.get_by_role("textbox", name="Name", exact=True)).to_have_value("Simulator")
        draft = second.get_by_role("textbox", name="Name", exact=True)
        draft.fill("Closing session")
        first.get_by_role("textbox", name="Name", exact=True).fill("Opening session")
        first.get_by_role("button", name="Save name").click()
        expect(first.get_by_role("heading", name="Opening session")).to_be_visible()
        save = second.get_by_role("button", name="Save name")
        save.click()
        expect(second.locator("#connection-edit-status")).to_contain_text("Opening session")
        expect(draft).to_have_value("Closing session")
        expect(second.get_by_role("button", name="Use saved name")).to_be_visible()
        second.get_by_role("button", name="Refresh connection").click()
        expect(save).to_be_enabled()
        expect(draft).to_have_value("Closing session")
        assert [req["expected_revision"] for req in requests] == [0, 0]
        save.click()
        expect(second.get_by_role("heading", name="Closing session")).to_be_visible()
        expect(second.locator("#connection-edit-status")).to_be_hidden()
        first.get_by_role("textbox", name="Name", exact=True).fill("Unsent draft")
        first.once("dialog", lambda dialog: dialog.accept())
        pause = first.get_by_role("button", name="Pause connection")
        pause.click()
        expect(first.locator("#connection-edit-status")).to_contain_text("again to confirm")
        assert connection["enabled"]
        first.get_by_role("button", name="Use saved name").click()
        expect(first.get_by_role("textbox", name="Name", exact=True)).to_have_value(
            "Closing session"
        )
        first.once("dialog", lambda dialog: dialog.accept())
        pause.click()
        expect(first.get_by_role("button", name="Resume connection")).to_be_visible()
        assert [req["expected_revision"] for req in requests] == [0, 0, 1, 1, 2]
    finally:
        context.close()
