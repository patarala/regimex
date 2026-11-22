"""Repeat and compare training runs from the shipped workspace."""

from copy import deepcopy
from typing import Any
from uuid import UUID

import pytest
from test_offline_report_browser import browser as browser
from test_training_workspace_browser import ORIGIN, RUN_ID, TrainingAPI

from regimex.directions import DEFAULT_ANALYSIS_PROFILE

OTHER_ID = str(UUID(int=15))


@pytest.mark.parametrize(
    "closed", [False, True], ids=["original-reservation", "closed-reservation"]
)
def test_copy_training_into_an_editable_draft(browser: Any, closed: bool) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.run["display_name"] = "Morning"
    source = api.run["training"]
    source.update(
        parameters={"sequence_key": 9007199254740993},
        training_runtime={"threads": 2},
        inference_runtime={"max_batch_size": 8},
        analysis_profile_id=DEFAULT_ANALYSIS_PROFILE,
    )
    if closed:
        api.benchmarks["development"] = ["replacement"]
        source["analysis_profile_id"] = "direction-32/2.1"
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    page.route(
        "**/training-options/benchmarks?*",
        lambda route: route.fulfill(
            json={
                "benchmark_ids": api.benchmarks["development"],
                "analysis_profile_id": DEFAULT_ANALYSIS_PROFILE,
            }
        ),
    )
    try:
        page.goto(f"{ORIGIN}/training/{RUN_ID}")
        page.get_by_text("Run details and settings", exact=True).click()
        settings = page.locator("#training-settings")
        expect(settings).to_contain_text("9007199254740993")
        expect(settings).to_contain_text('"architecture_id": "tcn"')
        expect(settings).not_to_contain_text("model_version_id")
        page.get_by_role("button", name="New run from this run").click()
        expect(page).to_have_url(f"{ORIGIN}/training/new")
        name = page.get_by_role("textbox", name="Run name (optional)")
        expect(name).to_have_value("Morning")
        benchmark = page.get_by_label("Reserved benchmark", exact=True)
        start = page.get_by_role("button", name="Start training")
        if closed:
            expect(start).to_be_disabled()
            expect(benchmark).to_have_value("")
            expect(page.locator("#training-prerequisite")).to_contain_text("Choose the missing")
            benchmark.select_option("replacement")
        expect(start).to_be_enabled()
        page.go_back()
        expect(page.locator("#training-detail-title")).to_have_text("Morning")
        page.go_forward()
        expect(benchmark).to_have_value("replacement" if closed else "selection-v1")
        name.fill("Afternoon")
        page.get_by_role("button", name="Edit JSON").click()
        editor = page.get_by_label("Run configuration (JSON)")
        assert "9007199254740993" in editor.input_value()
        assert not api.posts
        start.click()
        expect(page.locator("#training-detail-title")).to_have_text("Afternoon")
        [(path, submitted, _)] = api.posts
        assert path == "training-runs"
        assert submitted["parameters"] == {"sequence_key": 9007199254740993}
        assert submitted["policy"] == source["policy"]
        assert submitted["training_runtime"] == {"threads": 2}
        assert submitted["inference_runtime"] == {"max_batch_size": 8}
        assert submitted["benchmark_selection_id"] == ("replacement" if closed else "selection-v1")
        assert submitted["analysis_profile_id"] == DEFAULT_ANALYSIS_PROFILE
        assert "model_version_id" not in submitted
    finally:
        page.close()


@pytest.mark.parametrize("matched", [True, False], ids=["matched-profiles", "different-profiles"])
def test_compare_two_finished_runs_from_history_and_bookmark(browser: Any, matched: bool) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.run["display_name"] = "Control"
    api.run["trainable_parameters"] = 35011
    api.run["statistics"] = {"elapsed_seconds": 90, "fit_seconds": 80}
    api.run["training"].update(
        analysis_profile_id=DEFAULT_ANALYSIS_PROFILE, parameters={"hash": 9007199254740992}
    )
    challenger = deepcopy(api.run)
    challenger.update(run_id=OTHER_ID, display_name="Candidate", trainable_parameters=77091)
    challenger["training"]["parameters"]["hash"] = 9007199254740993
    challenger["output"] = {"validation_segment_macro_f1": 0.745}
    if not matched:
        challenger["training"]["analysis_profile_id"] = "direction-32/2.1"
    api.runs = [api.run, challenger]
    calls: list[str] = []
    page = browser.new_page(viewport={"width": 320, "height": 720})
    page.route(ORIGIN + "/**", api.dispatch)

    def score(route: Any) -> None:
        calls.append(route.request.url)
        route.fulfill(json={"model_version_id": "control", "validation_segment_macro_f1": 0.625})

    page.route("**/validation-score", score)
    try:
        page.goto(f"{ORIGIN}/training/new")
        page.get_by_role("button", name="Run history", exact=True).click()
        page.get_by_role("button", name="Select runs").click()
        compare = page.get_by_role("button", name="Compare", exact=True)
        expect(compare).to_be_disabled()
        page.get_by_role("checkbox", name="Select Control").check()
        page.get_by_role("checkbox", name="Select Candidate").check()
        compare.click()
        url = f"{ORIGIN}/training/compare?left={RUN_ID}&right={OTHER_ID}"
        expect(page).to_have_url(url)
        expect(page.get_by_role("heading", name="Compare runs")).to_be_focused()
        results = page.get_by_role("table", name="Run results")
        expect(results.get_by_role("row", name="Model fit time")).to_contain_text("1 min 20 s")
        diff = page.get_by_role("table", name="Settings that differ")
        expect(diff.get_by_role("row", name="parameters.hash").get_by_role("cell")).to_have_text(
            ["9007199254740992", "9007199254740993"]
        )
        note = page.locator("#training-comparison-status")
        if matched:
            expect(note).to_contain_text("+0.1200")
            expect(results.get_by_role("row", name="Validation interval macro-F1")).to_contain_text(
                "0.7450"
            )
            assert calls == [f"{ORIGIN}/api/v1/training-runs/{RUN_ID}/validation-score"]
        else:
            expect(note).to_contain_text(
                "requires the same dataset, validation labels and analysis profile"
            )
            expect(results.get_by_role("row", name="Validation interval macro-F1")).to_have_count(0)
            assert not calls
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        results.get_by_role("link", name="A · Control").click()
        expect(page.locator("#training-detail-title")).to_have_text("Control")
        page.go_back()
        expect(page).to_have_url(url)
        page.reload()
        expect(diff).to_be_visible()
        assert not api.posts
    finally:
        page.close()


def test_pending_runs_can_be_compared_without_archiving_or_reading_scores(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.run["job"]["state"] = "queued"
    api.run["display_name"] = "Pending"
    second = deepcopy(api.run)
    second.update(run_id=OTHER_ID, display_name="Interrupted")
    second["job"]["state"] = "cancelled"
    api.runs = [api.run, second]
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(f"{ORIGIN}/training/new")
        page.get_by_role("button", name="Select runs").click()
        page.get_by_role("checkbox", name="Select Pending").check()
        expect(page.get_by_role("button", name="Archive selected")).to_be_disabled()
        page.get_by_role("checkbox", name="Select Interrupted").check()
        expect(page.get_by_role("button", name="Archive selected (1 run)")).to_be_enabled()
        page.get_by_role("button", name="Compare", exact=True).click()
        expect(page.get_by_role("row", name="Status")).to_contain_text("Cancelled")
        expect(page.locator("#training-comparison-status")).to_contain_text(
            "Both runs must succeed"
        )
        assert not any(path.endswith("/validation-score") for path in api.pages)
        assert not api.archive_requests
    finally:
        page.close()
