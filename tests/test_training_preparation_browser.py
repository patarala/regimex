"""Prepare the first training dataset through the live browser, API and worker."""

from typing import Any
from urllib.parse import urlsplit
from uuid import UUID

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

    # Lose the first generation and reservation acknowledgements after the API commits.
    # The worker still runs the actual queue jobs, including sealed reference generation.
    jobs: dict[str, UUID] = {}
    tasks: list[dict[str, Any]] = []
    acknowledged: list[str] = []
    requested: list[str] = []

    def generate(route: Any) -> None:
        if route.request.method != "POST":
            route.continue_()
            return
        key = route.request.headers["idempotency-key"]
        response = route.fetch()
        assert response.status == 202, response.text()
        job_id = UUID(response.json()["run_id"])
        if key in jobs:
            assert jobs[key] == job_id
        else:
            jobs[key] = job_id
            tasks.append(route.request.post_data_json)
            run = ctx.service.run_admin(job_id)
            assert run is not None and run.operation.state == "succeeded", run
            if len(jobs) == 1:
                route.abort("connectionreset")
                return
        route.fulfill(response=response)

    def reserve(route: Any) -> None:
        response = route.fetch()
        assert response.status == 200, response.text()
        acknowledged.append(response.json()["benchmark_id"])
        if len(acknowledged) == 1:
            route.abort("connectionreset")
        else:
            route.fulfill(response=response)

    with tcp_server(app) as server:
        context = browser.new_context(viewport={"width": 320, "height": 720})
        page = context.new_page()
        second_tab = context.new_page()
        page.on("request", lambda req: requested.append(urlsplit(req.url).path))
        context.route("**/api/v1/datasets", generate)
        context.route("**/api/v1/reference-sets", generate)
        context.route("**/api/v1/training-options/selection", reserve)
        context.route(
            "**/api/v1/admin-runs/*",
            lambda route: route.fulfill(status=503, json={"detail": "Temporary failure"}),
            times=1,
        )
        try:
            page.goto(server.origin + "/training/new")
            page.locator("#token").fill(ctx.token)
            page.locator("#login-submit").click()
            page.get_by_text("Prepare training data", exact=True).click()
            for field in ("Development experiments", "Final dataset experiments"):
                expect(page.get_by_label(field)).to_have_attribute("max", "12")
                expect(page.get_by_label(field)).to_have_value("12")
            page.get_by_label("Development experiments").fill("13")
            page.get_by_role("button", name="Generate datasets").click()
            assert page.get_by_label("Development experiments").evaluate(
                "field => field.validity.rangeOverflow"
            )
            assert not jobs
            page.get_by_label("Development experiments").fill("9")
            page.get_by_label("Final dataset experiments").fill("5")
            page.get_by_label("Steps per experiment").fill("128")
            expect(page.locator("#preparation-counts")).to_have_text(
                "6 training and 2 validation experiments. "
                "3 separate experiments are reserved for final testing."
            )
            second_tab.goto(server.origin + "/training/new")
            second_tab.get_by_text("Prepare training data", exact=True).click()
            page.evaluate(
                """() => new Promise(locked => navigator.locks.request(
                  'regimex.training-preparation.v1.fixture-administrator',
                  () => new Promise(done => { window.releaseStart = done; locked(); })))"""
            )
            page.get_by_role("button", name="Generate datasets").click()
            second_tab.get_by_role("button", name="Generate datasets").click()
            page.evaluate("window.releaseStart()")
            expect(page.locator("#preparation-status")).to_contain_text(
                "Could not connect", timeout=60_000
            )
            page.reload()
            expect(second_tab.get_by_role("button", name="Continue preparation")).to_be_visible()
            page.get_by_role("button", name="Continue preparation").click()
            check = page.get_by_role("checkbox", name="I confirm")
            expect(check).to_be_visible(timeout=120_000)
            expect(check).not_to_be_checked()
            page.get_by_text("Review 14 source namespaces").click()
            expect(page.locator("#preparation-sources")).to_be_visible()
            assert len(page.locator("#preparation-sources").inner_text().splitlines()) == 14
            assert len(jobs) == 5 and len({r["plan"]["seed"] for r in tasks[:2]}) == 2
            assert [r["partition"] for r in tasks[2:]] == ["train", "validation", "test"]
            assert tasks[2]["dataset_id"] == tasks[3]["dataset_id"] != tasks[4]["dataset_id"]
            assert not acknowledged
            check.check()
            page.get_by_role("button", name="Reserve benchmark").click()
            expect(page.locator("#preparation-status")).to_contain_text("Could not connect")
            page.get_by_role("button", name="Continue preparation").click()
            check.check()
            page.get_by_role("button", name="Reserve benchmark").click()
            expect(second_tab.get_by_role("button", name="Use prepared data")).to_be_visible()
            page.get_by_role("button", name="Use prepared data").click()
            expect(
                page.get_by_role("combobox", name="Development dataset", exact=True)
            ).to_have_value(tasks[2]["dataset_id"])
            assert len(acknowledged) == 2 and acknowledged[0] == acknowledged[1]
            expect(page.get_by_label("Reserved benchmark")).to_have_value(acknowledged[0])
            expect(page.get_by_role("button", name="Start training")).to_be_enabled()
            history = ctx.client.get("/api/v1/training-runs", headers=ctx.headers)
            assert history.status_code == 200 and history.json()["items"] == []
            assert not any(path.endswith("/annotations") for path in requested)
            assert not page.evaluate(
                "Object.keys(localStorage).some(key => key.includes('training-preparation'))"
            )
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            expect(second_tab.locator("#data-preparation")).not_to_have_attribute("open", "")
        finally:
            context.close()
