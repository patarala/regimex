"""Prepare the first training dataset through the live browser, API and worker."""

import re
from typing import Any
from urllib.parse import parse_qs, urlsplit
from uuid import UUID, uuid4

import pytest
from test_admin_api import AdministrativeAPI
from test_admin_api import administrative_api as administrative_api
from test_offline_report_browser import browser as browser
from test_simulator_gateway_integration import tcp_server
from test_training_workspace_browser import ORIGIN, TrainingAPI

from regimex.api import create_app
from regimex.application.auth import AuthService
from regimex.auth_contracts import AuthPolicy
from regimex.infrastructure.auth_repository import AuthRepository


@pytest.mark.parametrize(
    ("limit", "draft", "expected"),
    [
        pytest.param(None, (None, None), ("48", "24"), id="older-catalog"),
        pytest.param(12, ("7", "8"), ("7", "8"), id="valid-edits"),
        pytest.param(12, (None, "13"), ("12", "13"), id="edit-over-limit"),
    ],
)
def test_late_catalog_updates_only_untouched_preparation_sizes(
    browser: Any,
    limit: int | None,
    draft: tuple[str | None, str | None],
    expected: tuple[str, str],
) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    page = browser.new_page()
    catalog: list[Any] = []
    page.route(ORIGIN + "/**", TrainingAPI().dispatch)
    page.route("**/api/v1/datasets?*", lambda route: catalog.append(route))
    try:
        with page.expect_request("**/api/v1/datasets?*"):
            page.goto(ORIGIN + "/training/new")
        page.get_by_text("Prepare training data", exact=True).click()
        sizes = [
            page.get_by_label("Development experiments"),
            page.get_by_label("Final dataset experiments"),
        ]
        action = page.get_by_role("button", name="Generate datasets")
        expect(action).to_be_disabled()
        for field, value in zip(sizes, draft, strict=True):
            expect(field).to_be_editable()
            if value is not None:
                field.fill(value)
        response: dict[str, Any] = {"items": [], "limit": 100, "offset": 0}
        if limit is not None:
            response["max_generation_experiments"] = limit
        catalog[0].fulfill(json=response)
        expect(action).to_be_enabled()
        for field, value in zip(sizes, expected, strict=True):
            expect(field).to_have_value(value)
            expect(field).to_have_attribute("max", str(limit or 1000))
            assert field.evaluate("input => input.validity.rangeOverflow") == (
                limit is not None and int(value) > limit
            )
    finally:
        page.close()


def test_resumable_preparation_generates_and_reserves_data_without_starting_training(
    administrative_api: AdministrativeAPI, browser: Any
) -> None:
    pytest.importorskip("numpy")
    expect = pytest.importorskip("playwright.sync_api").expect
    ctx = administrative_api
    ctx.service.settings = ctx.service.settings.model_copy(
        update={"max_generation_experiments": 12}
    )

    def app(origin: str) -> Any:
        return create_app(
            ctx.analysis,
            admin_service=ctx.service,
            benchmark_service=ctx.benchmark,
            auth_service=AuthService(
                AuthRepository(ctx.analysis.repository),
                AuthPolicy(public_origin=origin, allow_loopback_http=True),
            ),
        )

    def login(page: Any, url: str) -> None:
        page.goto(url)
        page.locator("#token").fill(ctx.token)
        page.locator("#login-submit").click()

    with tcp_server(app) as server:
        first = browser.new_context()
        try:
            page = first.new_page()
            login(page, server.origin + "/training/new")
            page.get_by_text("Prepare training data", exact=True).click()
            expect(page.get_by_label("Development experiments")).to_have_attribute("max", "12")
            page.get_by_label("Development experiments").fill("9")
            page.get_by_label("Final dataset experiments").fill("5")
            page.get_by_label("Steps per experiment").fill("128")
            expect(page.locator("#preparation-counts")).to_have_text(
                "6 training and 2 validation experiments. "
                "3 separate experiments are reserved for final testing."
            )
            page.get_by_role("button", name="Generate datasets").click()
            expect(page).to_have_url(re.compile(r"/training/new\?preparation="))
            permalink = page.url
        finally:
            first.close()

        # Advance only through jobs already accepted by the server, without an open browser.
        (preparation_id,) = parse_qs(urlsplit(permalink).query)["preparation"]
        url = f"/api/v1/training-preparations/{preparation_id}"
        for stage in ("development", "final", "train", "validation", "test"):
            accepted = ctx.client.get(url, headers=ctx.headers).json()
            assert accepted["stage"] == stage and accepted["state"] == "queued"
            completed = ctx.service.run_admin(UUID(accepted["operation_id"]))
            assert completed is not None and completed.operation.state == "succeeded", completed
        ready = ctx.client.get(url, headers=ctx.headers).json()
        assert len(ready["outputs"]) == 5 and ready["benchmark_id"] is None

        other = browser.new_context(viewport={"width": 320, "height": 720})
        try:
            page = other.new_page()
            login(page, server.origin + "/training/new")
            page.get_by_text("Prepare training data", exact=True).click()
            page.get_by_text("Saved preparations", exact=True).click()
            page.get_by_role("link", name=re.compile("Ready for review")).click()
            expect(page).to_have_url(permalink)
            check = page.get_by_role("checkbox", name="I confirm")
            expect(check).to_be_visible()
            expect(check).not_to_be_checked()
            page.get_by_text("Review 14 source namespaces").click()
            assert len(page.locator("#preparation-sources").inner_text().splitlines()) == 14
            check.check()

            def lose_acknowledgement(route: Any) -> None:
                response = route.fetch()
                assert response.status == 200, response.text()
                route.abort("connectionreset")

            page.route("**/training-options/selection", lose_acknowledgement, times=1)
            page.get_by_role("button", name="Reserve benchmark").click()
            expect(page.locator("#preparation-status")).to_contain_text("Could not connect")
            page.reload()
            page.get_by_role("button", name="Use prepared data").click()
            expect(page).to_have_url(server.origin + "/training/new")
            expect(
                page.get_by_role("combobox", name="Development dataset", exact=True)
            ).to_have_value(ready["outputs"]["development"])
            benchmark = ctx.client.get(url, headers=ctx.headers).json()["benchmark_id"]
            expect(page.get_by_label("Reserved benchmark")).to_have_value(benchmark)
            expect(page.get_by_role("button", name="Start training")).to_be_enabled()
            page.go_back()
            expect(page).to_have_url(permalink)
            expect(page.get_by_role("button", name="Use prepared data")).to_be_visible()
            assert (
                ctx.client.get("/api/v1/training-runs", headers=ctx.headers).json()["items"] == []
            )
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        finally:
            other.close()


def test_preparation_retry_is_scoped_and_keeps_completed_stages(
    administrative_api: AdministrativeAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from regimex.infrastructure.preparation_repository import PreparationRepository
    from regimex.service_worker import recover_operations

    ctx = administrative_api
    resource = "/api/v1/training-preparations"
    path = resource + "/" + str(uuid4())
    payload = {"development": 3, "final": 3, "steps": 64}
    ctx.service.settings = ctx.service.settings.model_copy(
        update={"max_generation_experiments": 5, "max_admin_pending_per_owner": 1}
    )
    assert (
        ctx.client.put(path, json={**payload, "development": 8}, headers=ctx.headers).status_code
        == 429
    )
    response = ctx.client.put(path, json=payload, headers=ctx.headers)
    assert response.status_code == 202, response.text
    first = response.json()
    assert ctx.client.put(path, json=payload, headers=ctx.headers).json() == first
    assert (
        ctx.client.put(path, json={**payload, "steps": 128}, headers=ctx.headers).status_code == 409
    )
    other = ctx.analysis.repository.create_token("separate-owner", "admin")
    foreign = {"Authorization": "Bearer " + other}
    assert ctx.client.get(path, headers=foreign).status_code == 404
    assert ctx.client.get(resource, headers=foreign).json()["items"] == []
    assert (
        ctx.client.get(path, headers={"Authorization": "Bearer " + ctx.user_token}).status_code
        == 403
    )
    with monkeypatch.context() as old_worker:
        old_worker.setattr(PreparationRepository, "advance", lambda *_: None)
        completed = ctx.service.run_admin(UUID(first["operation_id"]))
    assert completed is not None and completed.operation.state == "succeeded"
    competing = ctx.client.put(resource + "/" + str(uuid4()), json=payload, headers=ctx.headers)
    assert competing.status_code == 202, competing.text
    assert recover_operations(ctx.service.settings, ctx.analysis) == 0
    owner = ctx.analysis.authenticate(ctx.token)
    ctx.analysis.cancel_operation(owner, UUID(competing.json()["operation_id"]))
    assert recover_operations(ctx.service.settings, ctx.analysis) == 1
    assert recover_operations(ctx.service.settings, ctx.analysis) == 0
    second = ctx.client.get(path, headers=ctx.headers).json()
    assert second["stage"] == "final" and list(second["outputs"]) == ["development"]
    retry = {"operation_id": second["operation_id"]}
    assert ctx.client.post(path + "/retry", json=retry, headers=ctx.headers).status_code == 409
    ctx.analysis.cancel_operation(owner, UUID(second["operation_id"]))
    assert ctx.client.get(path, headers=ctx.headers).json()["state"] == "cancelled"
    assert ctx.client.post(path + "/retry", json=retry, headers=foreign).status_code == 404
    restored = ctx.client.post(path + "/retry", json=retry, headers=ctx.headers).json()
    assert restored["stage"] == second["stage"] and restored["outputs"] == second["outputs"]
    assert restored["state"] == "queued" and restored["operation_id"] != retry["operation_id"]
    assert ctx.client.post(path + "/retry", json=retry, headers=ctx.headers).json() == restored


def test_adopt_started_browser_preparation_reuses_its_jobs(
    administrative_api: AdministrativeAPI,
) -> None:
    ctx = administrative_api
    preparation = uuid4()
    path = f"/api/v1/training-preparations/{preparation}"
    request = {
        "development": 3,
        "final": 4,
        "steps": 64,
        "browser_seeds": {"development": 20, "final": 21},
    }
    started = []
    for stage, count, seed, fractions in (
        ("development", 3, 20, (0.7, 0.25)),
        ("final", 4, 21, (0.001, 0.001)),
    ):
        response = ctx.client.post(
            "/api/v1/datasets",
            json={
                "mode": "generate",
                "plan": {"experiment_count": count, "step_count": 64, "seed": seed},
                "splits": {
                    "seed": 1729,
                    "train_fraction": fractions[0],
                    "validation_fraction": fractions[1],
                },
            },
            headers={**ctx.headers, "Idempotency-Key": f"{preparation}:{stage}:0"},
        )
        assert response.status_code == 202, response.text
        started.append(response.json()["run_id"])
        if stage == "development":
            completed = ctx.service.run_admin(UUID(started[0]))
            assert completed is not None and completed.operation.state == "succeeded"
    mismatch = ctx.client.put(path, json={**request, "development": 4}, headers=ctx.headers)
    assert mismatch.status_code == 409
    assert ctx.client.get(path, headers=ctx.headers).status_code == 404
    adopted = ctx.client.put(path, json=request, headers=ctx.headers)
    assert adopted.status_code == 202, adopted.text
    assert adopted.json()["operation_id"] == started[1]
    assert list(adopted.json()["outputs"]) == ["development"]
    ctx.service.run_admin(UUID(started[1]))
    current = ctx.client.get(path, headers=ctx.headers).json()
    assert current["stage"] == "train" and set(current["outputs"]) == {"development", "final"}
    assert ctx.client.put(path, json=request, headers=ctx.headers).json() == current
