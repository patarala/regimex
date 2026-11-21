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
        self.benchmarks = {"development": ["selection-v1"]}
        self.metrics: dict[str, Any] = {
            "selected_seed": 23,
            "validation_segment_macro_f1": 0.72,
            "history": {
                "epochs": [
                    {
                        "seed": 11,
                        "epoch": 1,
                        "training_loss": 0.91,
                        "validation_segment_macro_f1": 0.65,
                    },
                    {
                        "seed": 23,
                        "epoch": 1,
                        "training_loss": 0.56,
                        "validation_segment_macro_f1": 0.72,
                    },
                ]
            },
        }

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
        if path == "training-options/benchmarks":
            dataset = parse_qs(url.query)["dataset_id"][0]
            route.fulfill(json={"benchmark_ids": self.benchmarks[dataset]})
            return
        if path == "reference-sets":
            assert parse_qs(url.query)["dataset_id"][0] in self.benchmarks
        if path == "datasets":
            assert parse_qs(url.query)["development_only"] == ["true"]
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
                    {"dataset_id": dataset, "partitions": {"train": 4, "validation": 2}}
                    for dataset in self.benchmarks
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
            f"training-runs/{RUN_ID}/metrics": self.metrics,
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
        page.get_by_role("link", name="Training", exact=True).press("Enter")
        expect(page.get_by_text("No training runs yet.")).to_be_visible()
        page.get_by_role("button", name="New run", exact=True).press("Enter")
        expect(page.get_by_label("Training labels")).to_have_value("training-labels")
        expect(page.get_by_label("Training labels").locator("option")).to_have_count(1)
        expect(page.get_by_label("Validation references")).to_have_value("validation-labels")
        expect(page.get_by_label("Validation references").locator("option")).to_have_count(1)
        expect(page.get_by_label("Development dataset").locator("option")).to_have_count(1)
        expect(page.get_by_label("Architecture").locator("option")).to_have_text(
            ["Temporal convolutional network (TCN)", "worker-verified"]
        )
        start = page.get_by_role("button", name="Start training")
        expect(page.get_by_label("Selection benchmark")).to_have_value("selection-v1")
        expect(start).to_be_enabled()
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
        expect(page.get_by_role("heading", name="TCN training")).to_be_visible()
        assert api.posts[0] == api.posts[1]
        assert api.posts[0][:2] == ("training-runs", expected_intake)

        expect(page.locator("#training-score")).to_contain_text("0.72")
        expect(page.get_by_role("link", name="Download metric data")).to_have_attribute(
            "href", f"/api/v1/training-runs/{RUN_ID}/metrics"
        )
        seed = page.get_by_label("Chart seed")
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
        page.get_by_role("link", name="Analyses", exact=True).press("Enter")
        expect(page.locator("#workspace")).to_be_visible()
        page.get_by_role("link", name="Training", exact=True).press("Enter")
        expect(page.get_by_text("This candidate is active.")).to_be_visible()
        assert len(api.posts) == 4
        assert script_errors == []
    finally:
        page.close()


@pytest.mark.parametrize("width", [320, 390])
def test_phone_run_history_collapses_on_selection_and_preserves_keyboard_access(
    browser: Any, width: int
) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    for index in range(13):
        row = completed_run(str(UUID(int=100 + index)), "training")
        row["job"]["state"] = "failed"
        api.runs.append(row)
    page = browser.new_page(viewport={"width": width, "height": 800})
    page.route(ORIGIN + "/**", api.dispatch)
    toggle = page.get_by_role("button", name="Run history", exact=True)
    new = page.get_by_role("button", name="New run", exact=True)
    rows = page.locator("#training-runs button")
    try:
        page.goto(ORIGIN + "/#training/new")
        expect(page.get_by_role("heading", name="New training run")).to_be_in_viewport()
        expect(toggle).to_have_attribute("aria-expanded", "false")
        expect(page.locator("#training-history")).to_be_hidden()
        expect(new).to_be_in_viewport()
        toggle.press("Space")
        expect(rows).to_have_count(12)
        expect(toggle).to_have_attribute("aria-expanded", "true")
        page.keyboard.press("Tab")
        expect(new).to_be_focused()
        page.keyboard.press("Tab")
        expect(rows.first).to_be_focused()
        page.get_by_role("button", name="Older").press("Enter")
        expect(rows).to_have_count(1)
        rows.first.press("Enter")
        heading = page.get_by_role("heading", name="TCN training")
        expect(heading).to_be_focused()
        expect(toggle).to_have_attribute("aria-expanded", "false")
        expect(page.locator("#training-history")).to_be_hidden()
        expect(page).to_have_url(ORIGIN + "/#training/" + api.runs[-1]["run_id"])
        page.set_viewport_size({"width": 1280, "height": 800})
        expect(toggle).to_be_hidden()
        expect(rows.first).to_be_visible()
        page.set_viewport_size({"width": width, "height": 800})
        toggle.press("Enter")
        rows.first.press("Enter")
        expect(heading).to_be_focused()
        expect(toggle).to_have_attribute("aria-expanded", "false")
        new.press("Enter")
        expect(page.get_by_role("heading", name="New training run")).to_be_focused()
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
        expect(page.get_by_role("link", name="Training", exact=True)).to_be_hidden()
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
        page.get_by_role("link", name="Training", exact=True).click()
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
        page.get_by_role("link", name="Training", exact=True).click()
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
        page.get_by_role("link", name="Training", exact=True).click()
        cancel = page.get_by_role("button", name="Cancel training")
        cancel.click()
        expect(page.locator("#training-state")).to_have_text("Cancelled")
        expect(cancel).to_be_hidden()
        expect(page.locator("#training-metrics")).to_be_hidden()
        assert api.cancellations == [f"jobs/{RUN_ID}/cancel"]
    finally:
        page.close()


def test_metrics_appear_when_training_finishes_and_seed_changes_stay_local(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.run["job"].update(state="queued", stage="queued", attempt=0, max_attempts=2)
    api.runs.append(api.run)
    metrics_path = f"training-runs/{RUN_ID}/metrics"
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    navigations: list[str] = []
    page.on("domcontentloaded", lambda: navigations.append(page.url))
    try:
        page.goto(ORIGIN)
        page.get_by_role("link", name="Training", exact=True).click()
        state = page.locator("#training-state")
        expect(state).to_have_text("Queued")
        expect(page.get_by_role("heading", name="Training metrics")).to_be_hidden()
        assert metrics_path not in api.pages

        api.run["job"].update(state="running", stage="training", attempt=1)
        expect(state).to_have_text("Running", timeout=7000)
        assert metrics_path not in api.pages

        page.route(
            f"**/{metrics_path}",
            lambda route: route.fulfill(status=503, json={"detail": "temporarily unavailable"}),
            times=1,
        )
        api.run["job"]["state"] = "succeeded"
        loss = page.get_by_role("img", name="Training loss by epoch")
        expect(page.locator("#training-sync")).to_contain_text(
            "Retrying automatically", timeout=7000
        )
        expect(loss).to_be_visible(timeout=7000)
        expect(loss.locator("circle title")).to_have_text("Epoch 1: 0.56")
        page.get_by_label("Chart seed").select_option("11")
        expect(loss.locator("circle title")).to_have_text("Epoch 1: 0.91")
        expect(page.locator("#training-score")).to_contain_text("seed 23")
        assert api.pages.count(metrics_path) == 1
        assert len(navigations) == 1
    finally:
        page.close()


def test_benchmark_options_follow_the_selected_development_dataset(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.benchmarks["other"] = ["other-selection", "repeat-selection"]
    api.benchmarks["waiting"] = []
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(ORIGIN)
        page.get_by_role("link", name="Training", exact=True).click()
        page.get_by_role("button", name="New run", exact=True).click()
        dataset = page.get_by_label("Development dataset")
        benchmark = page.get_by_label("Selection benchmark")
        start = page.get_by_role("button", name="Start training")
        expect(benchmark).to_have_value("selection-v1")
        expect(start).to_be_enabled()
        dataset.select_option("other")
        expect(benchmark.locator("option")).to_have_text(["other-selection", "repeat-selection"])
        benchmark.select_option("repeat-selection")

        held: list[Any] = []
        page.route(
            "**/training-options/benchmarks?dataset_id=waiting", lambda route: held.append(route)
        )
        with page.expect_request(lambda req: "benchmarks?dataset_id=waiting" in req.url):
            dataset.select_option("waiting")
        expect(start).to_be_disabled()
        assert len(held) == 1
        held[0].fulfill(json={"benchmark_ids": []})
        expect(benchmark).to_have_value("")
        expect(page.locator("#training-prerequisite")).to_contain_text(
            "This dataset has no benchmark open for model selection."
        )
        expect(page.get_by_role("link", name="Start benchmark selection")).to_be_visible()
        expect(start).to_be_disabled()
        retry = "**/training-options/benchmarks?dataset_id=development"
        page.route(retry, lambda request: request.abort("connectionreset"))
        dataset.select_option("development")
        expect(start).to_be_disabled()
        retry_button = page.get_by_role("button", name="Retry")
        expect(retry_button).to_be_visible()
        page.unroute(retry)
        retry_button.press("Enter")
        expect(benchmark).to_have_value("selection-v1")
        expect(start).to_be_enabled()
        expect(page.locator("#training-prerequisite")).to_be_hidden()

        with page.expect_request(lambda req: "benchmarks?dataset_id=waiting" in req.url):
            dataset.select_option("waiting")
        expect(start).to_be_disabled()
        dataset.select_option("other")
        expect(benchmark).to_have_value("other-selection")
        with page.expect_request_finished(lambda req: "benchmarks?dataset_id=waiting" in req.url):
            held[-1].fulfill(json={"benchmark_ids": []})
        expect(benchmark).to_have_value("other-selection")
        expect(start).to_be_enabled()
    finally:
        page.close()


def test_sparse_charts_align_and_all_values_are_keyboard_accessible_on_a_phone(
    browser: Any,
) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.runs.append(api.run)
    api.metrics["history"] = {
        "updates": [
            {
                "update": update,
                "training_loss": 0.0001034 / update,
                "validation_segment_macro_f1": 0.72 if update == 23 else None,
            }
            for update in range(1, 24)
        ]
    }
    page = browser.new_page(viewport={"width": 320, "height": 800})
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(ORIGIN)
        page.get_by_role("link", name="Training", exact=True).click()
        loss = page.get_by_role("img", name="Training loss by update")
        validation = page.get_by_role("img", name="Validation interval macro-F1 by update")
        # Both curves must use the full update range, even with a single validation checkpoint.
        expect(validation.locator("circle")).to_be_visible()
        last_loss = loss.locator("circle").last.bounding_box()
        last_validation = validation.locator("circle").bounding_box()
        assert last_loss and last_validation
        assert last_loss["x"] == pytest.approx(last_validation["x"], abs=1)
        first_loss = loss.locator("circle").first.bounding_box()
        assert first_loss and last_loss["x"] - first_loss["x"] > 100
        label = loss.locator("text").first.bounding_box()
        assert label and label["height"] >= 11
        assert loss.evaluate("""svg => {
            const area = svg.getBoundingClientRect();
            return [...svg.querySelectorAll('text')].every(label => {
                const box = label.getBoundingClientRect();
                return box.left >= area.left && box.right <= area.right;
            });
        }""")
        page.get_by_text("Metric values", exact=True).press("Enter")
        table = page.get_by_role("table", name="Recorded metric values by update")
        expect(table.get_by_role("row")).to_have_count(21)
        expect(table.get_by_role("row").nth(1)).to_contain_text("Not recorded")
        page.get_by_role("button", name="Next values").press("Enter")
        expect(page.get_by_role("button", name="Previous values")).to_be_focused()
        expect(table.get_by_role("row")).to_have_count(4)
        expect(table.get_by_role("row").last).to_contain_text("0.72")
        page.get_by_role("button", name="Previous values").press("Enter")
        expect(table.get_by_role("rowheader").first).to_have_text("1")
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    finally:
        page.close()


def test_saved_baseline_link_reopens_the_regularization_chart(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.run["training"]["architecture_id"] = "baseline"
    api.runs.append(api.run)
    api.metrics["history"] = {
        "candidates": [
            {"regularization": x, "validation_segment_macro_f1": y}
            for x, y in [(0.001, 0.2), (0.1, 0.72), (10, 0.5)]
        ],
    }
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(f"{ORIGIN}/#training/{RUN_ID}")
        heading = page.get_by_role("heading", name="Baseline training")
        expect(heading).to_be_visible()
        chart = page.get_by_role("img", name="Validation interval macro-F1 by regularization")
        expect(chart).to_be_visible()
        markers = chart.locator("circle")
        expect(markers).to_have_count(3)
        positions = markers.evaluate_all(
            "(nodes) => nodes.map(node => node.getBoundingClientRect().x)"
        )
        assert 0 < positions[0] < positions[1] < positions[2]
        assert positions[1] - positions[0] == pytest.approx(positions[2] - positions[1], abs=1)
        expect(chart.get_by_text("Regularization (log scale)", exact=True)).to_be_visible()
        expect(page.get_by_label("Chart seed")).to_be_hidden()
        expect(page.get_by_role("img", name="Training loss")).to_be_hidden()
        page.reload()
        expect(heading).to_be_visible()
        expect(chart).to_be_visible()
        page.get_by_role("button", name="New run", exact=True).click()
        new_run = page.get_by_role("heading", name="New training run")
        expect(new_run).to_be_visible()
        page.locator("#training-form-settings summary").click()
        runtime = page.get_by_label("Training runtime overrides")
        runtime.fill('{"threads": 2}')
        page.go_back()
        expect(heading).to_be_visible()
        expect(chart).to_be_visible()
        page.go_forward()
        expect(new_run).to_be_visible()
        expect(page.get_by_role("button", name="Start training")).to_be_enabled()
        expect(runtime).to_have_value('{"threads": 2}')
    finally:
        page.close()


def test_opening_training_preserves_report_and_comparison_history(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    first, second = str(UUID(int=31)), str(UUID(int=32))
    analyses = {
        analysis_id: {
            "analysis_id": analysis_id,
            "snapshot_id": "shared-snapshot",
            "job": {
                "state": "succeeded",
                "created_at": "2026-01-01T12:00:00Z",
                "model": {"model_version_id": "tcn"},
            },
            "instrument_ids": ["ETH/USD"],
            "step_count": 12,
            "result_available": True,
            "report_available": True,
        }
        for analysis_id in (first, second)
    }

    def analysis_response(route: Any) -> None:
        url = urlsplit(route.request.url)
        if url.path.endswith("/report/preview"):
            route.fulfill(content_type="text/html", body="<p>Price chart</p>")
        elif url.path.endswith("/comparison/preview"):
            assert parse_qs(url.query)["other"] == [second]
            route.fulfill(content_type="text/html", body="<p>Interval comparison</p>")
        elif url.path.endswith("/analyses"):
            route.fulfill(json={"items": list(analyses.values())})
        else:
            route.fulfill(json=analyses[url.path.rsplit("/", 1)[-1]])

    page = browser.new_page()
    page.route(ORIGIN + "/**", TrainingAPI().dispatch)
    page.route("**/api/v1/analyses**", analysis_response)
    try:
        report_url = f"{ORIGIN}/?analysis={first}"
        comparison_url = f"{report_url}&compare={second}"
        page.goto(report_url)
        report = page.frame_locator("#report-preview").get_by_text("Price chart")
        comparison = page.frame_locator("#comparison-preview").get_by_text("Interval comparison")
        expect(report).to_be_visible()
        page.get_by_text("Compare with another analysis", exact=True).click()
        page.get_by_role("button", name="Compare", exact=True).click()
        expect(comparison).to_be_visible()
        expect(page).to_have_url(comparison_url)
        page.get_by_role("link", name="Training", exact=True).click()
        expect(page.get_by_role("heading", name="Model training")).to_be_visible()

        page.go_back()
        expect(comparison).to_be_visible()
        page.go_back()
        expect(report).to_be_visible()
        expect(page).to_have_url(report_url)
        page.go_forward()
        expect(comparison).to_be_visible()
        expect(page).to_have_url(comparison_url)
        page.reload()
        expect(comparison).to_be_visible()
    finally:
        page.close()
