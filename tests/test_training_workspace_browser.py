"""Exercise the administrator's model lifecycle through the shipped browser assets."""

import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

import pytest
from test_offline_report_browser import browser as browser

import regimex

ASSETS = Path(regimex.__file__).parent / "web"
ORIGIN = "https://regimex.test"
RUN_ID = str(UUID(int=1))
EVALUATION_ID = str(UUID(int=2))
PUBLICATION_ID = str(UUID(int=3))
CANDIDATE = "model-browser-candidate"
PREVIOUS = "model-already-active"
POLICY = {"optimizer": "adamw", "epochs": 2, "seeds": [11, 23]}
SPEC = {
    "dataset_id": "development",
    "architecture_id": "tcn",
    "labels_reference_set_id": "training-labels",
    "validation_reference_set_id": "validation-labels",
    "benchmark_selection_id": "selection-v1",
    "parameters": {},
    "policy": POLICY,
}


def completed_run(run_id: str, kind: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "job": {"created_at": "2026-01-01T12:00:00Z", "kind": kind, "state": "succeeded"},
        "training": {**SPEC, "model_version_id": CANDIDATE} if kind == "training" else None,
    }


class TrainingAPI:
    def __init__(self, role: str = "admin") -> None:
        self.run = completed_run(RUN_ID, "training")
        self.runs: list[dict[str, Any]] = []
        self.actions: list[dict[str, Any]] = []
        self.posts: list[tuple[str, dict[str, Any], str]] = []
        self.cancellations: list[str] = []
        self.active_model = PREVIOUS
        self.pages: list[str] = []
        self.role = role

    def dispatch(self, route: Any) -> None:
        url = urlsplit(route.request.url)
        if url.path == "/" or url.path.startswith("/assets/"):
            asset = ASSETS / ("index.html" if url.path == "/" else Path(url.path).name)
            route.fulfill(path=asset, content_type=mimetypes.guess_type(asset)[0])
            return
        path = url.path.removeprefix("/api/v1/")
        self.pages.append(path)
        if route.request.method == "POST":
            self.post(route, path)
            return
        if path == "training-runs":
            params = parse_qs(url.query)
            start = int(params["offset"][0])
            limit = int(params["limit"][0])
            route.fulfill(json={"items": self.runs[start : start + limit]})
            return
        if path == "reference-sets":
            assert parse_qs(url.query)["dataset_id"] == ["development"]
        rows = [
            {"seed": seed, "epoch": 1, "training_loss": loss, "validation_segment_macro_f1": score}
            for seed, loss, score in ((11, 0.91, 0.65), (23, 0.56, 0.72))
        ]
        resources = {
            "session": {
                "principal": {"owner_id": "operator", "role": self.role},
                "csrf_token": "test",
            },
            "models": {
                "active_model_version_id": self.active_model,
                "items": [{"model_version_id": self.active_model}],
            },
            "storage": {
                "used_bytes": 0,
                "reserved_bytes": 0,
                "available_bytes": 1024,
                "limit_bytes": 1024,
            },
            "deletions": {"items": []},
            "experiments": {"items": []},
            "datasets": {
                "items": [
                    {"dataset_id": "holdout", "partitions": {"test": 2}},
                    {"dataset_id": "development", "partitions": {"train": 4, "validation": 2}},
                ]
            },
            "architectures": {
                "items": [
                    {
                        "architecture_id": name,
                        "executable": executable,
                        "default_training_policy": POLICY,
                    }
                    for name, executable in (
                        ("metadata-only", False),
                        ("tcn", None),
                        ("worker-verified", True),
                    )
                ]
            },
            "reference-sets": {
                "items": [
                    {
                        "reference_set_id": name,
                        "partition": partition,
                        "kind": "automatic",
                        "current_training_profile": compatible,
                    }
                    for name, partition, compatible in (
                        ("other-profile", "train", False),
                        ("training-labels", "train", True),
                        ("validation-labels", "validation", True),
                        ("final-labels", "test", True),
                    )
                ]
            },
            f"training-runs/{RUN_ID}": self.run,
            f"training-runs/{RUN_ID}/actions": {"items": self.actions},
            f"training-runs/{RUN_ID}/metrics": {
                "selected_seed": 23,
                "validation_segment_macro_f1": 0.72,
                "history": {"epochs": rows},
            },
        }
        route.fulfill(json=resources[path])

    def post(self, route: Any, path: str) -> None:
        request = route.request
        assert request.headers["x-csrf-token"] == "test"
        if path == f"jobs/{RUN_ID}/cancel":
            self.cancellations.append(path)
            self.run["job"]["state"] = "cancelled"
            route.fulfill(json=self.run["job"])
            return
        key = request.headers["idempotency-key"]
        assert UUID(key).version == 4
        self.posts.append((path, request.post_data_json, key))
        if path == "training-runs":
            if len(self.posts) == 1:
                # The server accepted the run, but the response never reached the browser.
                self.runs.append(self.run)
                route.abort("connectionreset")
                return
            result = self.run
        elif path == "evaluations":
            result = completed_run(EVALUATION_ID, "evaluation")
            self.actions.append(result)
        else:
            assert path == f"models/{CANDIDATE}/publish"
            result = completed_run(PUBLICATION_ID, "publication")
            self.actions.append(result)
            self.active_model = CANDIDATE
        route.fulfill(status=202, json=result)


@pytest.mark.parametrize("width", [390, 1280], ids=["phone", "desktop"])
def test_admin_trains_evaluates_and_publishes_with_keyboard(browser: Any, width: int) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    page = browser.new_page(viewport={"width": width, "height": 900})
    script_errors: list[str] = []
    page.on("pageerror", lambda error: script_errors.append(str(error)))
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(ORIGIN)
        page.get_by_role("button", name="Training", exact=True).press("Enter")
        expect(page.get_by_text("No training runs yet.")).to_be_visible()
        page.get_by_role("button", name="New run", exact=True).press("Enter")
        expect(page.get_by_label("Training labels")).to_have_value("training-labels")
        expect(page.get_by_label("Training labels").locator("option")).to_have_count(1)
        expect(page.get_by_label("Validation references")).to_have_value("validation-labels")
        expect(page.get_by_label("Validation references").locator("option")).to_have_count(1)
        expect(page.get_by_label("Development dataset").locator("option")).to_have_count(1)
        expect(page.get_by_label("Architecture").locator("option")).to_have_text(
            ["tcn", "worker-verified"]
        )
        start = page.get_by_role("button", name="Start training")
        expect(start).to_be_disabled()
        page.get_by_label("Selection benchmark ID").fill("selection-v1")
        expected_intake = SPEC.copy()
        if width == 1280:
            page.locator("#training-form summary").click()
            runtime = page.get_by_label("Training runtime overrides")
            runtime.fill("[]")
            start.press("Enter")
            expect(runtime).to_be_focused()
            expect(page.locator("#training-notice")).to_contain_text("Enter a JSON object")
            assert not api.posts
            runtime.fill('{"time_budget_seconds": 60}')
            expected_intake["training_runtime"] = {"time_budget_seconds": 60}
        start.press("Enter")
        expect(page.locator("#training-notice")).to_contain_text("Could not connect")
        start.press("Enter")
        expect(page.get_by_role("heading", name="tcn training")).to_be_visible()
        assert api.posts[0] == api.posts[1]
        assert api.posts[0][:2] == ("training-runs", expected_intake)

        expect(page.locator("#training-score")).to_contain_text("0.72")
        expect(page.get_by_role("link", name="Download metric data")).to_have_attribute(
            "href", f"/api/v1/training-runs/{RUN_ID}/metrics"
        )
        seed = page.get_by_label("Training seed")
        expect(seed).to_have_value("23")
        loss = page.get_by_role("img", name="Training loss by epoch")
        expect(loss.locator("circle title")).to_have_text("Epoch 1: 0.56")
        seed.select_option("11")
        expect(loss.locator("circle title")).to_have_text("Epoch 1: 0.91")
        expect(
            page.get_by_role("img", name="Validation interval macro-F1 by epoch")
        ).to_be_visible()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")

        publish = page.get_by_role("button", name="Publish candidate")
        expect(publish).to_be_hidden()
        page.get_by_role("button", name="Evaluate candidate").press("Enter")
        expect(page.get_by_role("link", name="Evaluation JSON")).to_have_attribute(
            "href", f"/api/v1/evaluations/{EVALUATION_ID}/result"
        )
        assert api.posts[2][:2] == (
            "evaluations",
            {
                "model_version_id": CANDIDATE,
                "dataset_id": "development",
                "reference_set_id": "validation-labels",
                "partition": "validation",
            },
        )
        page.once("dialog", lambda dialog: dialog.dismiss())
        publish.press("Enter")
        expect(publish).to_be_enabled()
        assert len(api.posts) == 3
        page.once("dialog", lambda dialog: dialog.accept())
        publish.press("Enter")
        expect(page.get_by_text("This candidate is active.")).to_be_visible()
        assert api.posts[3][:2] == (
            f"models/{CANDIDATE}/publish",
            {
                "evaluation_run_id": EVALUATION_ID,
                "expected_previous": PREVIOUS,
            },
        )
        page.get_by_role("button", name="Back to analyses").press("Enter")
        expect(page.locator("#workspace")).to_be_visible()
        page.get_by_role("button", name="Training", exact=True).press("Enter")
        expect(page.get_by_text("This candidate is active.")).to_be_visible()
        assert len(api.posts) == 4
        assert script_errors == []
    finally:
        page.close()


def test_analysis_user_has_no_training_controls(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI(role="user")
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(ORIGIN)
        expect(page.locator("#workspace")).to_be_visible()
        expect(page.get_by_role("button", name="Training", exact=True)).to_be_hidden()
        assert "training-runs" not in api.pages
    finally:
        page.close()


@pytest.mark.parametrize("total", [12, 13])
def test_training_history_only_offers_another_page_when_one_exists(
    browser: Any, total: int
) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    for index in range(total):
        row = completed_run(str(UUID(int=100 + index)), "training")
        row["job"]["state"] = "failed"
        api.runs.append(row)
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(ORIGIN)
        page.get_by_role("button", name="Training", exact=True).click()
        rows = page.locator("#training-runs li")
        older = page.get_by_role("button", name="Older")
        newer = page.get_by_role("button", name="Newer")
        expect(rows).to_have_count(12)
        expect(newer).to_be_hidden()
        if total == 13:
            older.click()
            expect(rows).to_have_count(1)
            expect(rows.locator("button")).to_have_attribute("data-run-id", api.runs[-1]["run_id"])
            expect(older).to_be_hidden()
            newer.click()
            expect(rows).to_have_count(12)
            expect(older).to_be_visible()
        else:
            expect(older).to_be_hidden()
    finally:
        page.close()


def test_validation_and_publication_stay_locked_until_their_status_arrives(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.runs.append(api.run)
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(ORIGIN)
        page.get_by_role("button", name="Training", exact=True).click()
        evaluate = page.get_by_role("button", name="Evaluate candidate")
        publish = page.get_by_role("button", name="Publish candidate")
        expect(evaluate).to_be_enabled()
        pattern = f"**/training-runs/{RUN_ID}/actions?*"
        held: list[Any] = []

        def hold_status(route: Any) -> None:
            held.append(route)

        for button, kind in [(evaluate, "evaluation"), (publish, "publication")]:
            page.route(pattern, hold_status)
            if kind == "publication":
                page.once("dialog", lambda dialog: dialog.accept())
            with page.expect_request(
                lambda request: urlsplit(request.url).path.endswith("/actions")
            ):
                button.click()
            # The mutation returned 202, and the follow-up status request is still pending.
            expect(evaluate).to_be_disabled()
            if kind == "publication":
                expect(publish).to_be_disabled()
            pending_action = {
                **api.actions[-1],
                "job": {**api.actions[-1]["job"], "state": "running"},
            }
            assert len(held) == 1
            page.unroute(pattern)
            held.pop().fulfill(json={"items": [*api.actions[:-1], pending_action]})
            expect(page.locator("#training-actions")).to_contain_text("Running")
            expect(evaluate).to_be_disabled()
            page.get_by_role("button", name="Refresh", exact=True).click()
            if kind == "evaluation":
                expect(publish).to_be_enabled()
            else:
                expect(page.get_by_text("This candidate is active.")).to_be_visible()
        assert [path for path, _, _ in api.posts] == ["evaluations", f"models/{CANDIDATE}/publish"]
    finally:
        page.close()


def test_cancelling_running_training_updates_the_selected_run(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.run["job"].update(state="running", stage="training", attempt=1, max_attempts=2)
    api.runs.append(api.run)
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(ORIGIN)
        page.get_by_role("button", name="Training", exact=True).click()
        cancel = page.get_by_role("button", name="Cancel training")
        cancel.click()
        expect(page.locator("#training-state")).to_have_text("Cancelled")
        expect(cancel).to_be_hidden()
        expect(page.locator("#training-metrics")).to_be_hidden()
        assert api.cancellations == [f"jobs/{RUN_ID}/cancel"]
    finally:
        page.close()
