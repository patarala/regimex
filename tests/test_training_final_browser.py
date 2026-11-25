"""Exercise the final reference gate from a saved training run."""

from typing import Any
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

import pytest
from test_offline_report_browser import browser as browser
from test_training_workspace_browser import (
    CANDIDATE,
    EVALUATION_ID,
    ORIGIN,
    RUN_ID,
    TrainingAPI,
    completed_run,
    evaluation_report,
)

BENCHMARK = "market/held out + 1"
FINAL_ID = str(UUID(int=191))


def test_freeze_open_evaluate_and_publish_held_out_result(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.run["training"]["benchmark_selection_id"] = BENCHMARK
    api.runs.append(api.run)
    api.actions.append(completed_run(EVALUATION_ID, "evaluation"))
    view: dict[str, Any] = {
        "frozen_choice": None,
        "reservation": {"selection_started_at": "2026-01-01", "opened_at": None},
    }
    mutations: list[str] = []
    page = browser.new_page(viewport={"width": 390, "height": 844})
    page.route(ORIGIN + "/**", api.dispatch)

    def benchmark(route: Any) -> None:
        request = route.request
        url = urlsplit(request.url)
        assert parse_qs(url.query) == {"benchmark_id": [BENCHMARK]}
        if request.method == "GET":
            route.fulfill(json=view)
            return
        assert request.headers["x-csrf-token"] == "test"
        operation = url.path.rsplit("/", 1)[-1]
        mutations.append(operation)
        if operation == "freeze":
            assert request.post_data_json == {"model_version_id": CANDIDATE}
            view["frozen_choice"] = {"model_version_id": CANDIDATE}
        elif operation == "open":
            assert request.post_data is None
            assert view["frozen_choice"]
            view["reservation"]["opened_at"] = "2026-01-02"
        else:
            assert operation == "evaluations" and view["reservation"]["opened_at"]
            assert request.post_data_json == {"model_version_id": CANDIDATE}
            attempt = completed_run(FINAL_ID, "evaluation")
            attempt["final_benchmark_id"] = BENCHMARK
            attempt["job"]["state"] = "queued"
            api.actions.insert(0, attempt)
            # The worker accepted the job before the client lost its connection.
            route.abort("connectionreset")
            return
        route.fulfill(json=view)

    page.route("**/benchmarks?*", benchmark)
    page.route("**/benchmarks/*?*", benchmark)
    report = {
        **evaluation_report(),
        "scope": "controlled_final_benchmark",
        "benchmark_id": BENCHMARK,
        "experiment_count": 15,
        "segment_macro_f1": 0.45,
        "independent": False,
    }
    page.route(f"**/evaluations/{FINAL_ID}/result", lambda route: route.fulfill(json=report))
    try:
        page.goto(f"{ORIGIN}/training/{RUN_ID}")
        final = page.get_by_role("region", name="Final benchmark")
        freeze = final.get_by_role("button", name="Freeze this model")
        expect(freeze).to_be_enabled()
        page.once("dialog", lambda dialog: dialog.dismiss())
        freeze.click()
        assert not mutations
        page.once("dialog", lambda dialog: dialog.accept())
        freeze.click()
        unseal = final.get_by_role("button", name="Open held-out test")
        expect(unseal).to_be_enabled()
        assert mutations == ["freeze"]
        page.reload()
        expect(unseal).to_be_enabled()
        page.once("dialog", lambda dialog: dialog.dismiss())
        unseal.click()
        assert mutations == ["freeze"]
        page.once("dialog", lambda dialog: dialog.accept())
        unseal.click()
        final.get_by_role("button", name="Evaluate frozen model").click()
        expect(final).to_contain_text("Queued")
        page.reload()
        expect(final).to_contain_text("Queued")
        expect(final.get_by_role("button", name="Evaluate frozen model")).to_be_hidden()
        api.actions[0]["job"]["state"] = "succeeded"
        expect(final).to_contain_text("15 held-out experiments", timeout=10_000)
        expect(final).to_contain_text("Automatic references")
        expect(final).to_contain_text("0.45")
        expect(page.locator("#training-actions")).to_contain_text("3 validation experiments")
        expect(final.get_by_role("link", name="Download JSON")).to_have_attribute(
            "href", f"/api/v1/evaluations/{FINAL_ID}/result"
        )
        page.once("dialog", lambda dialog: dialog.accept())
        page.get_by_role("button", name="Use as default").click()
        expect(page.get_by_text("New analyses use this model by default.")).to_be_visible()
        assert api.posts[-1][1]["evaluation_run_id"] == FINAL_ID
        assert mutations == ["freeze", "open", "evaluations"]
    finally:
        page.close()


def test_benchmark_frozen_for_another_model_is_read_only(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.runs.append(api.run)
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    page.route(
        "**/benchmarks?*",
        lambda route: route.fulfill(
            json={
                "frozen_choice": {"model_version_id": "model-other"},
                "reservation": {"opened_at": "2026-01-02", "selection_started_at": "2026-01-01"},
            }
        ),
    )
    try:
        page.goto(f"{ORIGIN}/training/{RUN_ID}")
        final = page.get_by_role("region", name="Final benchmark")
        expect(final).to_contain_text("frozen model model-other")
        expect(final.get_by_role("button")).to_have_count(0)
        expect(page.get_by_role("button", name="Evaluate model")).to_be_enabled()
    finally:
        page.close()


def test_final_retry_key_survives_reloading_after_lost_response(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    final = completed_run(FINAL_ID, "evaluation")
    final["final_benchmark_id"] = api.run["training"]["benchmark_selection_id"]
    final["job"]["state"] = "failed"
    api.actions.append(final)
    keys: list[str] = []
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)

    def reject(route: Any) -> None:
        keys.append(route.request.headers["idempotency-key"])
        route.abort("connectionreset")

    page.route("**/benchmarks/evaluations?*", reject)
    try:
        page.goto(f"{ORIGIN}/training/{RUN_ID}")
        retry = page.get_by_role("button", name="Retry final evaluation")
        for _ in range(2):
            retry.click()
            expect(page.locator("#training-notice")).to_contain_text("Could not connect")
            page.reload()
        assert len(keys) == 2 and keys[0] == keys[1]
        final["run_id"] = str(UUID(int=192))
        page.reload()
        retry.click()
        expect(page.locator("#training-notice")).to_contain_text("Could not connect")
        assert keys[-1] != keys[0]
    finally:
        page.close()
