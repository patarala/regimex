"""Exercise the administrator's model lifecycle through the shipped browser assets."""

import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

import pytest
from test_offline_report_browser import browser as browser

import regimex
from regimex.model_registry import FixedUpdatePolicy, baseline_descriptor, tcn_descriptor

ASSETS = Path(regimex.__file__).parent / "web"
ORIGIN = "https://regimex.test"
RUN_ID = str(UUID(int=1))
EVALUATION_ID = str(UUID(int=2))
PUBLICATION_ID = str(UUID(int=3))
CANDIDATE = "model-browser-candidate"
PREVIOUS = "model-already-active"
TCN = tcn_descriptor().describe().model_dump(mode="json")
BASELINE = baseline_descriptor().describe().model_dump(mode="json")
POLICY = {**TCN["default_training_policy"], "epochs": 2, "seeds": [11, 23]}
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
        "display_name": None,
        "name_revision": 0,
        "job": {"created_at": "2026-01-01T12:00:00Z", "kind": kind, "state": "succeeded"},
        "training": {**SPEC, "model_version_id": CANDIDATE} if kind == "training" else None,
    }


def evaluation_report() -> dict[str, Any]:
    return {
        "evaluation_version": "2.4",
        "scope": "fixed_model_validation",
        "segment_macro_f1": 0.6,
        "point_macro_f1": 0.8,
        "experiment_count": 3,
        "instrument_sequence_count": 5,
        "reference_kind": "automatic",
        "point_metrics": {"defined": True},
        "segment_metrics": {
            "iou_threshold": 0.5,
            "class_names": ["sideways", "growth", "decline"],
            "per_class": {
                "sideways": {"support": 3, "predicted": 2, "matched": 2, "f1": 0.8},
                "growth": {"support": 1, "predicted": 1, "matched": 1, "f1": 1.0},
                "decline": {"support": 1, "predicted": 0, "matched": 0, "f1": 0.0},
            },
        },
    }


class TrainingAPI:
    def __init__(self, role: str = "admin") -> None:
        self.run = completed_run(RUN_ID, "training")
        self.runs: list[dict[str, Any]] = []
        self.actions: list[dict[str, Any]] = []
        self.posts: list[tuple[str, dict[str, Any], str]] = []
        self.drop_first_training_response = False
        self.cancellations: list[str] = []
        self.active_model = PREVIOUS
        self.pages: list[str] = []
        self.progress_rows: list[dict[str, Any]] = []
        self.progress_kind = "epochs"
        self.progress_page_size = 512
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
        if path == f"training-runs/{RUN_ID}/progress":
            params = parse_qs(url.query)
            attempt = self.run["job"]["attempt"]
            start = int(params["after"][0]) if int(params["attempt"][0]) == attempt else 0
            end = min(start + self.progress_page_size, len(self.progress_rows))
            rows = self.progress_rows[start:end]
            route.fulfill(
                json={
                    "attempt": attempt,
                    "cursor": end,
                    "has_more": end < len(self.progress_rows),
                    "history": {self.progress_kind: rows} if rows else {},
                }
            )
            return
        if path == "training-options/benchmarks":
            dataset = parse_qs(url.query)["dataset_id"][0]
            route.fulfill(json={"benchmark_ids": self.benchmarks[dataset]})
            return
        train_ref, validation_ref = "training-labels", "validation-labels"
        if path == "reference-sets":
            dataset = parse_qs(url.query)["dataset_id"][0]
            assert dataset in self.benchmarks
            if dataset != "development":
                train_ref, validation_ref = f"{dataset}-train", f"{dataset}-validation"
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
                    {**TCN, "architecture_id": "metadata-only", "executable": False},
                    {**TCN, "default_training_policy": POLICY},
                    BASELINE,
                    {**TCN, "architecture_id": "worker-verified", "executable": True},
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
                        (train_ref, "train", True),
                        (validation_ref, "validation", True),
                        ("final-labels", "test", True),
                    )
                ]
            },
            f"training-runs/{RUN_ID}": self.run,
            f"training-runs/{RUN_ID}/actions": {"items": self.actions},
            f"training-runs/{RUN_ID}/metrics": self.metrics,
            f"evaluations/{EVALUATION_ID}/result": evaluation_report(),
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
                training = request.post_data_json.copy()
                self.run["display_name"] = training.pop("display_name", None)
                self.run["training"].update(training)
                self.runs.append(self.run)
                if self.drop_first_training_response:
                    # The server accepted the run, but the response never reached the browser.
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


@pytest.mark.parametrize("width", [320, 1280], ids=["phone", "desktop"])
def test_admin_trains_evaluates_and_publishes_with_keyboard(browser: Any, width: int) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.drop_first_training_response = True
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
        expect(page.get_by_label("Validation labels")).to_have_value("validation-labels")
        expect(page.get_by_label("Validation labels").locator("option")).to_have_count(1)
        expect(page.get_by_label("Development dataset").locator("option")).to_have_count(1)
        expect(page.get_by_label("Architecture").locator("option")).to_have_text(
            ["Temporal convolutional network (TCN)", "Logistic baseline", "worker-verified"]
        )
        start = page.get_by_role("button", name="Start training")
        expect(page.get_by_label("Reserved benchmark")).to_have_value("selection-v1")
        expect(start).to_be_enabled()
        expected_intake = SPEC.copy()
        title = "TCN · development"
        if width == 1280:
            title = "Morning volatility"
            page.get_by_label("Run name (optional)").fill(title)
            expected_intake["display_name"] = title
            page.get_by_text("Advanced settings", exact=True).click()
            runtime = page.get_by_label("Training resources")
            runtime.fill("[]")
            start.press("Enter")
            expect(runtime).to_be_focused()
            expect(page.locator("#training-notice")).to_contain_text("Enter a JSON object")
            assert not api.posts
            runtime.fill('{"time_budget_seconds": 60}')
            expected_intake["training_runtime"] = {"time_budget_seconds": 60}
        start.press("Enter")
        expect(page.locator("#training-notice")).to_contain_text("Could not connect")
        if width == 1280:
            page.get_by_role("button", name="Edit JSON").press("Enter")
            expect(page.get_by_label("Run configuration (JSON)")).to_be_visible()
        start.press("Enter")
        expect(page.get_by_role("heading", name=title)).to_be_visible()
        assert api.posts[0] == api.posts[1]
        assert api.posts[0][:2] == ("training-runs", expected_intake)

        expect(page.locator("#training-score")).to_contain_text("0.72")
        expect(page.get_by_role("link", name="Download metric data")).to_have_attribute(
            "href", f"/api/v1/training-runs/{RUN_ID}/metrics"
        )
        seed = page.get_by_label("Training repeat")
        expect(seed).to_have_value("23")
        expect(seed.locator("option:checked")).to_have_text("Seed 23 (saved model)")
        loss = page.get_by_role("img", name="Training loss by epoch")
        expect(loss.locator("circle title")).to_have_text("Epoch 1: 0.56")
        seed.select_option("11")
        expect(loss.locator("circle title")).to_have_text("Epoch 1: 0.91")
        expect(
            page.get_by_role("img", name="Validation interval macro-F1 by epoch")
        ).to_be_visible()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")

        publish = page.get_by_role("button", name="Use as default")
        expect(publish).to_be_hidden()
        page.get_by_role("button", name="Evaluate model").press("Enter")
        results = page.locator(".evaluation-result")
        path = f"evaluations/{EVALUATION_ID}/result"
        expect(page.get_by_role("button", name="Evaluate model")).to_be_hidden()
        expect(results.get_by_role("link", name="Download JSON")).to_have_attribute(
            "href", f"/api/v1/{path}"
        )
        expect(results).to_contain_text("3 validation experiments")
        expect(results.locator("dt")).to_have_text(["Interval macro-F1", "Per-step macro-F1"])
        expect(results.locator("dd")).to_have_text(["0.6", "0.8"])
        intervals = results.get_by_role("table", name="Intervals by direction")
        expect(intervals.get_by_role("columnheader")).to_have_text(
            ["Metric", "Sideways", "Growth", "Decline"]
        )
        expect(intervals.get_by_role("rowheader")).to_have_text(
            ["F1", "Reference", "Predicted", "Matched"]
        )
        for label, values in [
            ("F1", ["0.8", "1", "0"]),
            ("Reference", ["3", "1", "1"]),
            ("Predicted", ["2", "1", "0"]),
            ("Matched", ["2", "1", "0"]),
        ]:
            row = intervals.get_by_role("row").filter(
                has=page.get_by_role("rowheader", name=label, exact=True)
            )
            expect(row.get_by_role("cell")).to_have_text(values)
        assert intervals.evaluate("table => table.offsetWidth <= table.parentElement.clientWidth")
        results.get_by_text("All fields", exact=True).press("Enter")
        expect(results.locator("pre")).to_contain_text('"segment_macro_f1": 0.6')
        results.get_by_text("All fields", exact=True).press("Enter")
        expect(intervals).to_be_visible()
        assert api.pages.count(path) == 1
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
        expect(page.get_by_text("New analyses use this model by default.")).to_be_visible()
        expect(intervals).to_be_visible()
        assert api.pages.count(path) == 1
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
        expect(page.get_by_text("New analyses use this model by default.")).to_be_visible()
        assert len(api.posts) == 4
        assert script_errors == []
    finally:
        page.close()


def test_hovered_training_help_floats_without_moving_fields(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    page.route(ORIGIN + "/**", TrainingAPI().dispatch)
    try:
        page.goto(ORIGIN + "/#training/new")
        button = page.get_by_role("button", name="Help for Reserved benchmark")
        field = page.get_by_label("Reserved benchmark", exact=True)
        note = page.locator("#" + str(field.get_attribute("aria-describedby")))
        submit = page.get_by_role("button", name="Start training")
        expect(submit).to_be_enabled()
        position = "el => el.getBoundingClientRect().top + scrollY"
        initial_positions = [control.evaluate(position) for control in (field, submit)]

        button.hover()
        expect(note).to_be_visible()
        expect(note).to_contain_text("Its test labels stay sealed")
        assert [control.evaluate(position) for control in (field, submit)] == initial_positions
        page.clock.install()
        note.hover()
        # Let the pointer's grace period for crossing the gap expire while reading the note.
        page.clock.run_for(1000)
        expect(note).to_be_visible()
        page.get_by_role("heading", name="New training run").hover()
        page.clock.run_for(1000)
        expect(note).to_be_hidden()

        button.click()
        page.get_by_role("heading", name="New training run").hover()
        page.clock.run_for(1000)
        expect(note).to_be_visible()
        name = page.get_by_label("Run name (optional)")
        name.click()
        expect(note).to_be_hidden()
        expect(name).to_be_focused()
    finally:
        page.close()


def test_training_help_opens_on_keyboard_focus_and_survives_form_updates(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(ORIGIN + "/#training/new")
        page.get_by_text("Training policy", exact=True).press("Enter")
        field = page.get_by_label("Checks without improvement", exact=True)
        button = page.get_by_role("button", name="Help for Checks without improvement")
        before = field.input_value()
        note = page.locator("#" + str(field.get_attribute("aria-describedby")))
        field.focus()
        page.keyboard.press("Shift+Tab")
        expect(button).to_be_focused()
        expect(note).to_contain_text("validation interval macro-F1")
        expect(note).to_be_visible()
        page.keyboard.press("Escape")
        expect(note).to_be_hidden()
        expect(button).to_be_focused()

        page.keyboard.press("Tab")
        page.keyboard.press("Shift+Tab")
        expect(note).to_be_visible()
        page.keyboard.press("Tab")
        expect(field).to_be_focused()
        expect(note).to_be_hidden()
        page.keyboard.press("Shift+Tab")
        page.keyboard.press("Space")
        page.keyboard.press("Tab")
        expect(field).to_be_focused()
        expect(note).to_be_visible()
        page.keyboard.press("Escape")
        expect(note).to_be_hidden()
        expect(button).to_be_focused()
        expect(field).to_have_value(before)

        button.press("Enter")
        expect(note).to_be_visible()
        page.get_by_label("Architecture", exact=True).select_option("baseline")
        expect(note).to_be_hidden()
        seed_button = page.get_by_role("button", name="Help for Random seed", exact=True)
        seed_button.focus()
        seed = page.get_by_label("Random seed", exact=True)
        seed_note = page.locator("#" + str(seed.get_attribute("aria-describedby")))
        expect(seed_note).to_contain_text("logistic baseline")
        expect(seed_note).to_be_visible()
        assert not api.posts
    finally:
        page.close()


def test_training_help_on_phone_stays_onscreen_and_dismisses_on_tap(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    page = browser.new_page(viewport={"width": 320, "height": 440}, has_touch=True)
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(ORIGIN + "/#training/new")
        button = page.get_by_role("button", name="Help for Reserved benchmark")
        field = page.get_by_label("Reserved benchmark", exact=True)
        note = page.locator("#" + str(field.get_attribute("aria-describedby")))
        button.evaluate(
            "el => scrollTo(0, scrollY + el.getBoundingClientRect().bottom - innerHeight + 6)"
        )
        expect(note).to_be_hidden()
        button.tap()
        expect(note).to_be_visible()
        expect(button).to_have_attribute("aria-expanded", "true")
        bounds = note.bounding_box()
        assert bounds is not None
        assert 0 <= bounds["x"] < bounds["x"] + bounds["width"] <= 320
        assert 0 <= bounds["y"] < bounds["y"] + bounds["height"] <= 440
        anchor = button.bounding_box()
        assert anchor is not None
        assert (
            bounds["x"] <= anchor["x"] + anchor["width"]
            and bounds["x"] + bounds["width"] >= anchor["x"]
        )
        assert abs(bounds["y"] + bounds["height"] - anchor["y"]) <= 20
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        button.tap()
        expect(note).to_be_hidden()
        button.tap()
        expect(note).to_be_visible()
        page.get_by_role("heading", name="New training run").tap()
        expect(note).to_be_hidden()
        assert not api.posts
    finally:
        page.close()


def test_epoch_policy_uses_catalog_defaults_and_focuses_the_invalid_field(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(ORIGIN + "/#training/new")
        policy = page.get_by_text("Training policy", exact=True)
        policy.press("Enter")
        budget = page.get_by_label("Training budget")
        expect(budget).to_have_value("epochs")
        epochs = page.get_by_label("Maximum epochs", exact=True)
        expect(epochs).to_have_value("2")
        seeds = page.get_by_label("Random seeds", exact=True)
        assert [int(value) for value in seeds.input_value().split(",")] == POLICY["seeds"]
        epochs.fill("12")
        seeds.fill("5, 7")
        batch_size = page.get_by_label("Batch size")
        batch_size.fill("0")
        policy.press("Enter")
        expect(batch_size).to_be_hidden()
        start = page.get_by_role("button", name="Start training")
        start.press("Enter")
        expect(batch_size).to_be_focused()
        assert batch_size.evaluate("field => field.validity.rangeUnderflow")
        assert not api.posts

        batch_size.fill("8")
        page.get_by_label("Learning rate").fill("0.01")
        page.get_by_label("Weight decay").fill("0")
        page.get_by_label("Epochs between validation checks").fill("3")
        page.get_by_label("Checks without improvement").fill("2")
        start.press("Enter")
        expect(page.get_by_role("heading", name="TCN · development")).to_be_visible()
        assert api.posts[0][1]["policy"] == {
            **POLICY,
            "epochs": 12,
            "seeds": [5, 7],
            "batch_size": 8,
            "learning_rate": 0.01,
            "weight_decay": 0,
            "validation_interval": 3,
            "patience": 2,
        }
    finally:
        page.close()


@pytest.mark.parametrize("custom", [False, True], ids=["automatic", "custom"])
def test_update_budget_submits_only_update_settings(browser: Any, custom: bool) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(ORIGIN + "/#training/new")
        page.get_by_text("Training policy", exact=True).press("Enter")
        page.get_by_label("Training budget").select_option("updates")
        for label in (
            "Maximum epochs",
            "Checks without improvement",
            "Epochs between validation checks",
        ):
            expect(page.get_by_label(label, exact=True)).to_be_hidden()
        page.get_by_label("Updates per seed").fill("3")
        checkpoints = page.get_by_label("Validation checkpoints")
        start = page.get_by_role("button", name="Start training")
        if custom:
            checkpoints.fill("1, 2")
            start.press("Enter")
            expect(checkpoints).to_be_focused()
            expect(checkpoints).to_have_attribute("aria-invalid", "true")
            assert not api.posts
            checkpoints.fill("1, 3")
        else:
            expect(checkpoints).to_have_value("")

        start.press("Enter")
        expect(page.get_by_role("heading", name="TCN · development")).to_be_visible()
        expected = FixedUpdatePolicy(
            seeds=(11, 23), updates=3, validation_updates=(1, 3) if custom else (1, 2, 3)
        )
        assert api.posts[0][1]["policy"] == expected.model_dump(mode="json")
    finally:
        page.close()


@pytest.mark.parametrize("submit_in_json", [False, True], ids=["use-form", "start-in-json"])
def test_pasting_a_policy_preserves_the_rest_of_the_run(browser: Any, submit_in_json: bool) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.drop_first_training_response = not submit_in_json
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(ORIGIN + "/#training/new")
        page.get_by_label("Run name (optional)").fill("Checkpoint run")
        page.get_by_role("button", name="Edit JSON").click()
        editor = page.get_by_label("Run configuration (JSON)")
        policy = FixedUpdatePolicy(
            seeds=(4,),
            updates=6,
            validation_updates=(2, 6),
            learning_rate=0.025,
        ).model_dump(mode="json", exclude_unset=True)
        editor.fill(json.dumps(policy))
        start = page.get_by_role("button", name="Start training")
        if not submit_in_json:
            start.click()
            expect(page.locator("#training-notice")).to_contain_text("Could not connect")
            page.get_by_role("button", name="Use form").click()
            expect(editor).to_be_hidden()
            page.get_by_text("Training policy", exact=True).click()
            expect(page.get_by_label("Training budget")).to_have_value("updates")
            expect(page.get_by_label("Updates per seed")).to_have_value("6")
            expect(page.get_by_label("Learning rate")).to_have_value("0.025")
            expect(page.get_by_label("Random seeds", exact=True)).to_have_value("4")
        start.click()
        expect(page.get_by_role("heading", name="Checkpoint run")).to_be_visible()
        assert api.posts[0][1] == {**SPEC, "display_name": "Checkpoint run", "policy": policy}
        if not submit_in_json:
            assert api.posts[0] == api.posts[1]
    finally:
        page.close()


def test_run_json_returns_to_the_form_with_the_selected_dataset_and_references(
    browser: Any,
) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.benchmarks["prices"] = ["prices-selection"]
    page = browser.new_page(viewport={"width": 320, "height": 800})
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(ORIGIN + "/#training/new")
        start = page.get_by_role("button", name="Start training")
        expect(start).to_be_enabled()
        edit = page.get_by_role("button", name="Edit JSON")
        edit.click()
        editor = page.get_by_label("Run configuration (JSON)")
        assert json.loads(editor.input_value()) == SPEC
        editor.fill('{"policy":')
        start.click()
        expect(editor).to_be_focused()
        expect(editor).to_have_value('{"policy":')
        assert not api.posts

        request = {
            **SPEC,
            "architecture_id": "baseline",
            "dataset_id": "prices",
            "display_name": "Price sensitivity",
            "analysis_profile_id": "prices-v2",
            "benchmark_selection_id": "prices-selection",
            "labels_reference_set_id": "prices-train",
            "validation_reference_set_id": "prices-validation",
            "policy": {**BASELINE["default_training_policy"], "seeds": [8]},
            "parameters": {"regularization": [0.5, 2]},
            "training_runtime": {"threads": 1},
            "inference_runtime": {"max_batch_size": 32},
        }
        use_form = page.get_by_role("button", name="Use form")
        invalid = json.dumps({**request, "validation_reference_set_id": "unknown-reference"})
        editor.fill(invalid)
        use_form.click()
        expect(editor).to_be_focused()
        expect(editor).to_have_value(invalid)
        assert not api.posts

        editor.fill(json.dumps(request))
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        use_form.click()
        expect(editor).to_be_hidden()
        expect(page.get_by_label("Development dataset")).to_have_value("prices")
        expect(page.get_by_label("Training labels")).to_have_value("prices-train")
        expect(page.get_by_label("Validation labels")).to_have_value("prices-validation")
        expect(page.get_by_label("Reserved benchmark")).to_have_value("prices-selection")
        expect(page.get_by_label("Architecture")).to_have_value("baseline")
        edit.click()
        assert json.loads(editor.input_value()) == request
        request["training_runtime"] = {"memory_budget_bytes": 9007199254740993}
        editor.fill(json.dumps(request))
        use_form.click()
        expect(page.locator("#training-prerequisite")).to_contain_text("precision")
        expect(editor).to_be_focused()
        start.click()
        expect(page.get_by_role("heading", name="Price sensitivity")).to_be_visible()
        assert api.posts[0][1] == request
    finally:
        page.close()


def test_invalid_json_draft_survives_back_forward_until_discarded(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(ORIGIN + "/#training")
        page.get_by_role("button", name="New run", exact=True).click()
        name = page.get_by_label("Run name (optional)")
        name.fill("Draft before paste")
        page.get_by_role("button", name="Edit JSON").click()
        editor = page.get_by_label("Run configuration (JSON)")
        invalid = '{"optimizer": "adamw",'
        editor.fill(invalid)
        page.get_by_role("button", name="Use form").click()
        expect(editor).to_be_focused()
        page.go_back()
        expect(page.get_by_text("No training runs yet.")).to_be_visible()
        page.go_forward()
        expect(editor).to_be_visible()
        expect(editor).to_have_value(invalid)
        page.get_by_role("button", name="Discard JSON edits").click()
        expect(name).to_be_visible()
        expect(name).to_have_value("Draft before paste")
        page.get_by_role("button", name="Edit JSON").click()
        assert json.loads(editor.input_value()) == {**SPEC, "display_name": "Draft before paste"}
        assert not api.posts
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
        heading = page.get_by_role("heading", name="TCN · development")
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


@pytest.mark.parametrize("name", ["W" * 120, "سوق" * 40])
def test_long_run_names_fit_phone_history_and_detail(browser: Any, name: str) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.run["display_name"] = name
    api.runs.append(api.run)
    page = browser.new_page(viewport={"width": 320, "height": 820})
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(f"{ORIGIN}/#training/{RUN_ID}")
        title = page.get_by_role("heading", name=name)
        expect(title).to_be_visible()
        assert title.evaluate("el => el.clientWidth >= el.scrollWidth")
        page.get_by_role("button", name="Run history").click()
        row = page.locator("#training-runs button")
        expect(row).to_have_attribute("title", name)
        expect(row.locator("strong")).to_have_text(name)
        assert row.evaluate("el => el.clientWidth >= el.scrollWidth")
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    finally:
        page.close()


def test_run_names_survive_conflicts_and_older_list_responses(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    dataset_id = "dataset-12345678abcdef0012345678abcdef00"
    default_title = "TCN · Dataset 12345678"
    api.run["training"]["dataset_id"] = dataset_id
    api.runs.append(api.run)
    old_row = api.run.copy()
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    received: list[dict[str, Any]] = []

    def rename(route: Any) -> None:
        request = route.request
        assert request.method == "PATCH"
        assert request.headers["x-csrf-token"] == "test"
        update = request.post_data_json
        received.append(update)
        if update["expected_revision"] != api.run["name_revision"]:
            route.fulfill(status=409, json={"error": {"message": "The name was changed."}})
            return
        name = update["display_name"]
        api.run["display_name"] = name.strip() if name else None
        api.run["name_revision"] += 1
        route.fulfill(
            json={
                "run_id": RUN_ID,
                "display_name": api.run["display_name"],
                "revision": api.run["name_revision"],
            }
        )

    page.route(f"**/training-runs/{RUN_ID}/name", rename)
    try:
        page.goto(f"{ORIGIN}/#training/{RUN_ID}")
        heading = page.locator("#training-detail-title")
        row = page.locator("#training-runs button[aria-current=true]")
        expect(heading).to_have_text(default_title)
        expect(row.locator("strong")).to_have_text(default_title)
        page.get_by_text("Run details and settings").click()
        expect(page.locator("#training-facts")).to_contain_text(dataset_id)

        held: list[Any] = []
        page.route("**/training-runs?*", lambda route: held.append(route), times=1)
        with page.expect_request(lambda req: urlsplit(req.url).path.endswith("/training-runs")):
            page.get_by_role("button", name="Refresh", exact=True).click()
        page.get_by_role("button", name="Rename", exact=True).click()
        field = page.get_by_label("Run name", exact=True)
        save = page.get_by_role("button", name="Save name")
        draft = "  Opening Δ  "
        field.fill(draft)
        api.run.update(display_name="Another desk", name_revision=1)
        save.press("Enter")
        expect(heading).to_have_text("Another desk")
        expect(field).to_have_value(draft)
        expect(page.locator("#training-name-status")).to_contain_text("name changed elsewhere")
        save.press("Enter")
        expect(heading).to_have_text("Opening Δ")
        expect(row.locator(".training-run-context")).to_have_text(default_title)
        expect(page.get_by_role("button", name="Rename", exact=True)).to_be_focused()

        with page.expect_request_finished(
            lambda req: urlsplit(req.url).path.endswith("/training-runs")
        ):
            held[0].fulfill(json={"items": [old_row]})
        expect(row.locator("strong")).to_have_text("Opening Δ")
        expect(heading).to_have_text("Opening Δ")
        page.get_by_role("button", name="Rename", exact=True).click()
        field.fill("")
        save.press("Enter")
        expect(heading).to_have_text(default_title)
        expect(row.locator(".training-run-context")).to_have_count(0)
        assert received == [
            {"display_name": draft, "expected_revision": 0},
            {"display_name": draft, "expected_revision": 1},
            {"display_name": None, "expected_revision": 2},
        ]
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
        evaluate = page.get_by_role("button", name="Evaluate model")
        publish = page.get_by_role("button", name="Use as default")
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
            if kind == "evaluation":
                expect(evaluate).to_be_disabled()
            else:
                expect(evaluate).to_be_hidden()
                expect(publish).to_be_disabled()
            pending_action = {
                **api.actions[-1],
                "job": {**api.actions[-1]["job"], "state": "running"},
            }
            assert len(held) == 1
            page.unroute(pattern)
            held.pop().fulfill(json={"items": [*api.actions[:-1], pending_action]})
            expect(page.locator("#training-actions")).to_contain_text("Running")
            expect(page.locator("#training-evaluate")).to_be_hidden()
            page.get_by_role("button", name="Refresh", exact=True).click()
            if kind == "evaluation":
                expect(publish).to_be_enabled()
            else:
                expect(page.get_by_text("New analyses use this model by default.")).to_be_visible()
        assert [path for path, _, _ in api.posts] == ["evaluations", f"models/{CANDIDATE}/publish"]
    finally:
        page.close()


def test_evaluation_retries_without_resubmitting_and_ignores_old_responses(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.runs.append(api.run)
    api.actions.append(completed_run(EVALUATION_ID, "evaluation"))
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    path = f"/api/v1/evaluations/{EVALUATION_ID}/result"
    requests: list[Any] = []

    def send_report(route: Any) -> None:
        requests.append(route)
        if len(requests) == 2:
            route.fulfill(status=503, json={"error": {"message": "Storage unavailable."}})
        elif len(requests) == 3:
            route.fulfill(json=evaluation_report())

    page.route("**" + path, send_report)
    results = page.locator(".evaluation-result")
    try:
        with page.expect_request(lambda request: urlsplit(request.url).path == path):
            page.goto(ORIGIN + "/#training/" + RUN_ID)
        expect(results.get_by_role("status")).to_have_text("Loading evaluation…")
        page.get_by_role("button", name="New run", exact=True).click()
        with page.expect_request(lambda request: urlsplit(request.url).path == path):
            page.go_back()
        expect(results.get_by_role("status")).to_have_text("Storage unavailable.")
        with page.expect_request_finished(lambda request: urlsplit(request.url).path == path):
            requests[0].fulfill(json=evaluation_report())
        expect(results.get_by_role("table", include_hidden=True)).to_have_count(0)
        expect(results.get_by_role("link", name="Download JSON")).to_be_visible()
        expect(page.locator("#training-evaluate")).to_be_hidden()
        with page.expect_request(lambda request: urlsplit(request.url).path == path):
            results.get_by_role("button", name="Retry loading").press("Enter")
        expect(results.get_by_role("table", name="Intervals by direction")).to_be_visible()
        assert not api.posts
    finally:
        page.close()


@pytest.mark.parametrize("state", ["failed", "cancelled"])
def test_unsuccessful_evaluation_can_be_retried(browser: Any, state: str) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    attempt = completed_run(str(UUID(int=4)), "evaluation")
    attempt["job"]["state"] = state
    if state == "failed":
        attempt["job"]["failure"] = {"message": "Worker timed out."}
    api.actions.append(attempt)
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(f"{ORIGIN}/#training/{RUN_ID}")
        actions = page.locator("#training-actions")
        expect(actions).to_contain_text(state.capitalize())
        if state == "failed":
            expect(actions).to_contain_text("Worker timed out.")
        page.get_by_role("button", name="Retry evaluation").press("Enter")
        expect(actions.get_by_role("table", name="Intervals by direction")).to_be_visible()
        expect(page.locator("#training-evaluate")).to_be_hidden()
        expect(actions).not_to_contain_text(state.capitalize())
        assert [path for path, _, _ in api.posts] == ["evaluations"]
    finally:
        page.close()


def test_duplicate_evaluations_use_one_report_for_display_and_publication(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    # The actions API orders ties by descending ID.
    api.actions = [completed_run(str(UUID(int=i)), "evaluation") for i in (5, 4, 2)]
    api.actions[0]["job"]["state"] = "running"
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(f"{ORIGIN}/#training/{RUN_ID}")
        actions = page.locator("#training-actions")
        expect(actions.get_by_role("table", name="Intervals by direction")).to_be_visible()
        expect(actions.locator("li")).to_have_count(1)
        expect(page.locator("#training-evaluate")).to_be_hidden()
        publish = page.get_by_role("button", name="Use as default")
        page.once("dialog", lambda dialog: dialog.accept())
        publish.press("Enter")
        expect(page.get_by_text("New analyses use this model by default.")).to_be_visible()
        assert [path for path in api.pages if path.startswith("evaluations/")] == [
            f"evaluations/{EVALUATION_ID}/result"
        ]
        assert api.posts[0][1]["evaluation_run_id"] == EVALUATION_ID
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
        page.locator('summary[aria-label="Training loss values"]').press("Enter")
        page.get_by_label("Training repeat").select_option("11")
        expect(loss.locator("circle title")).to_have_text("Epoch 1: 0.91")
        table = page.get_by_role("table", name="Training loss by epoch")
        expect(table.get_by_role("cell")).to_have_text("0.91")
        expect(
            page.get_by_role("table", name="Validation interval macro-F1 by epoch")
        ).to_be_hidden()
        expect(page.locator("#training-score")).to_contain_text("seed 23")
        assert api.pages.count(metrics_path) == 1
        assert len(navigations) == 1
    finally:
        page.close()


def test_training_repeat_explains_early_stopping_live_and_after_completion(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.run["training"]["policy"] = {**POLICY, "epochs": 3, "patience": 1, "validation_interval": 1}
    api.run["job"].update(state="running", stage="training", attempt=1)
    first = {"seed": 11, "epoch": 1, "training_loss": 0.4, "validation_segment_macro_f1": 0.6}
    api.progress_rows = [first]
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(f"{ORIGIN}/#training/{RUN_ID}")
        ending = page.locator("#training-repeat-end")
        expect(page.get_by_role("img", name="Training loss by epoch")).to_be_visible()
        expect(ending).to_be_hidden()
        api.progress_rows.append({**first, "epoch": 2, "stop_reason": "early_stopping"})
        refresh = page.get_by_role("button", name="Refresh", exact=True)
        refresh.click()
        expect(ending).to_contain_text("Stopped early at epoch 2 of 3")
        expect(ending).to_contain_text("Interval macro-F1 did not improve for 1 validation check.")
        expect(page.locator("#training-state")).to_have_text("Running")

        # Published TCN histories omit live-only messages, including existing bundles.
        api.metrics["history"] = {
            "epochs": [
                first,
                {**first, "epoch": 2},
                *[
                    {
                        **first,
                        "seed": 23,
                        "epoch": epoch,
                        "validation_segment_macro_f1": 0.6 + 0.02 * epoch,
                    }
                    for epoch in range(1, 4)
                ],
            ]
        }
        api.metrics["repeat_outcomes"] = [
            {"seed": 11, "step": 2, "reason": "early_stopping"},
            {"seed": 23, "step": 3, "reason": "epoch_limit"},
        ]
        api.run["job"]["state"] = "succeeded"
        refresh.click()
        expect(page.locator("#training-state")).to_have_text("Complete")
        expect(ending).to_have_text("Completed all 3 epochs.")
        page.get_by_label("Training repeat").select_option("11")
        expect(ending).to_contain_text("Stopped early at epoch 2 of 3")
    finally:
        page.close()


def test_live_updates_preserve_the_selected_repeat_and_open_table(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.run["job"].update(state="running", stage="training", attempt=1)
    api.progress_kind = "updates"
    api.progress_page_size = 10

    def point(seed: int, update: int) -> dict[str, Any]:
        return {
            "seed": seed,
            "update": update,
            "training_loss": 1 / update,
            "validation_segment_macro_f1": 0.0 if update == 20 else None,
        }

    api.progress_rows = [point(11, update) for update in range(1, 22)]
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(f"{ORIGIN}/#training/{RUN_ID}")
        expect(page.get_by_text("1 of 21 updates evaluated")).to_be_visible()
        expect(page.locator("#training-score")).to_be_hidden()
        expect(page.get_by_role("link", name="Download metric data")).to_be_hidden()
        loss = page.get_by_role("img", name="Training loss by update")
        page.locator('summary[aria-label="Training loss values"]').press("Enter")
        page.get_by_role("button", name="Next values").press("Enter")
        previous = page.get_by_role("button", name="Previous values")
        expect(previous).to_be_focused()
        table = page.get_by_role("table", name="Training loss by update")
        expect(table.get_by_role("rowheader")).to_have_text(["21"])

        api.progress_rows.append(point(11, 22))
        expect(table.get_by_role("rowheader")).to_have_text(["21", "22"], timeout=7000)
        expect(previous).to_be_focused()
        expect(
            page.get_by_role("img", name="Validation interval macro-F1 by update").locator(
                "circle title"
            )
        ).to_have_text("Update 20: 0")

        api.progress_rows.append(point(23, 1))
        repeat = page.get_by_label("Training repeat")
        expect(repeat).to_have_value("23", timeout=7000)
        repeat.select_option("11")
        expect(table.get_by_role("rowheader").first).to_have_text("1")
        svg = loss.element_handle()
        assert svg
        api.progress_rows.append(point(23, 2))
        refresh = page.get_by_role("button", name="Refresh", exact=True)
        with page.expect_response(lambda r: "/progress?" in r.url):
            refresh.click()
        with page.expect_response(lambda r: "after=24" in r.url):
            refresh.click()
        expect(repeat).to_have_value("11")
        assert svg.evaluate("node => node.isConnected")

        api.metrics["history"] = {"updates": api.progress_rows}
        api.run["job"]["state"] = "succeeded"
        refresh.click()
        expect(page.locator("#training-score")).to_contain_text("saved model (seed 23)")
        expect(repeat).to_have_value("11")
        expect(table).to_be_visible()
        expect(page.get_by_role("link", name="Download metric data")).to_be_visible()
    finally:
        page.close()


def test_live_candidate_chart_recovers_from_errors_and_ignores_a_stale_attempt(
    browser: Any,
) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.run["job"].update(state="running", stage="training", attempt=1, max_attempts=2)
    api.run["training"]["architecture_id"] = "baseline"
    api.progress_kind = "candidates"
    api.progress_rows = [{"regularization": 1.0, "validation_segment_macro_f1": 0.5}]
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    progress_url = f"**/training-runs/{RUN_ID}/progress?*"
    page.route(progress_url, lambda route: route.fulfill(status=503, json={}), times=1)
    try:
        page.goto(f"{ORIGIN}/#training/{RUN_ID}")
        expect(page.locator("#training-progress-notice")).to_be_visible()
        chart = page.get_by_role("img", name="Validation interval macro-F1 by regularization")
        expect(chart.locator("circle title")).to_have_text("Regularization 1: 0.5", timeout=7000)
        expect(page.locator("#training-progress-notice")).to_be_hidden()
        held: list[Any] = []
        page.route(progress_url, lambda route: held.append(route), times=1)
        refresh = page.get_by_role("button", name="Refresh", exact=True)
        with page.expect_request(lambda req: "/progress?" in req.url):
            refresh.click()
        api.run["job"].update(state="queued", stage="retry_wait")
        refresh.click()
        expect(chart).to_be_hidden()
        api.run["job"].update(state="running", stage="training", attempt=2)
        api.progress_rows = [{"regularization": 0.1, "validation_segment_macro_f1": 0.8}]
        refresh.click()
        expect(chart.locator("circle title")).to_have_text("Regularization 0.1: 0.8")
        assert len(held) == 1
        held[0].fulfill(
            json={
                "attempt": 1,
                "cursor": 2,
                "has_more": False,
                "history": {
                    "candidates": [{"regularization": 10, "validation_segment_macro_f1": 0.9}]
                },
            }
        )
        expect(chart.locator("circle title")).to_have_text("Regularization 0.1: 0.8")
        api.run["job"]["state"] = "cancelled"
        refresh.click()
        expect(page.locator("#training-metrics")).to_be_hidden()
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
        benchmark = page.get_by_label("Reserved benchmark")
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
        expect(page.get_by_text("1 of 23 updates evaluated")).to_be_visible()
        page.locator('summary[aria-label="Validation interval macro-F1 values"]').press("Enter")
        scores = page.get_by_role("table", name="Validation interval macro-F1 by update")
        expect(scores.get_by_role("rowheader")).to_have_text(["23"])
        expect(scores.get_by_role("cell")).to_have_text(["0.72"])
        page.locator('summary[aria-label="Training loss values"]').press("Enter")
        losses = page.get_by_role("table", name="Training loss by update")
        expect(losses.get_by_role("row")).to_have_count(21)
        page.get_by_role("button", name="Next values").press("Enter")
        expect(page.get_by_role("button", name="Previous values")).to_be_focused()
        expect(losses.get_by_role("rowheader")).to_have_text(["21", "22", "23"])
        expect(scores.get_by_role("rowheader")).to_have_text(["23"])
        page.get_by_role("button", name="Previous values").press("Enter")
        expect(losses.get_by_role("rowheader").first).to_have_text("1")
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
        heading = page.get_by_role("heading", name="Baseline · development")
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
        expect(page.get_by_label("Training repeat")).to_be_hidden()
        expect(page.get_by_role("img", name="Training loss")).to_be_hidden()
        page.reload()
        expect(heading).to_be_visible()
        expect(chart).to_be_visible()
        page.get_by_role("button", name="New run", exact=True).click()
        expect(page.get_by_role("heading", name="New training run")).to_be_visible()
        page.go_back()
        expect(heading).to_be_visible()
        expect(chart).to_be_visible()
    finally:
        page.close()


def test_policy_drafts_survive_architecture_changes_and_browser_navigation(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(ORIGIN + "/#training")
        page.get_by_role("button", name="New run", exact=True).click()
        new_run = page.get_by_role("heading", name="New training run")
        expect(new_run).to_be_visible()
        page.get_by_text("Training policy", exact=True).click()
        page.get_by_text("Advanced settings", exact=True).click()
        parameters = page.get_by_label("Model configuration (JSON)")
        parameters.fill('{"channels": 8}')
        architecture = page.get_by_label("Architecture")
        epochs = page.get_by_label("Maximum epochs", exact=True)
        epochs.fill("8")
        architecture.select_option("baseline")
        seed = page.get_by_label("Random seed", exact=True)
        expect(seed).to_have_value("11")
        expect(parameters).to_have_value("{}")
        for label in (
            "Training budget",
            "Batch size",
            "Learning rate",
            "Weight decay",
            "Maximum epochs",
            "Epochs between validation checks",
            "Checks without improvement",
        ):
            expect(page.get_by_label(label, exact=True)).to_be_hidden()
        seed.fill("19")
        parameters.fill('{"context_sizes": [4, 16]}')
        architecture.select_option("tcn")
        expect(parameters).to_have_value('{"channels": 8}')
        expect(epochs).to_have_value("8")
        architecture.select_option("baseline")
        expect(seed).to_have_value("19")
        runtime = page.get_by_label("Training resources")
        runtime.fill('{"threads": 2}')
        name = page.get_by_label("Run name (optional)")
        name.fill("Weekend baseline")
        page.go_back()
        expect(page.get_by_text("No training runs yet.")).to_be_visible()
        page.go_forward()
        expect(new_run).to_be_visible()
        expect(page.get_by_role("button", name="Start training")).to_be_enabled()
        expect(runtime).to_have_value('{"threads": 2}')
        expect(name).to_have_value("Weekend baseline")
        expect(seed).to_be_visible()
        expect(seed).to_have_value("19")
        page.get_by_role("button", name="Start training").click()
        expect(page.get_by_role("heading", name="Weekend baseline")).to_be_visible()
        assert api.posts[0][1] == {
            **SPEC,
            "architecture_id": "baseline",
            "display_name": "Weekend baseline",
            "training_runtime": {"threads": 2},
            "parameters": {"context_sizes": [4, 16]},
            "policy": {**BASELINE["default_training_policy"], "seeds": [19]},
        }
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


def test_model_size_presets_edit_the_layout_and_identify_saved_runs(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.run.update(trainable_parameters=135555)
    api.runs.append(api.run)
    page = browser.new_page(viewport={"width": 320, "height": 800})
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(f"{ORIGIN}/#training/{RUN_ID}")
        expect(page.locator("#training-detail-title")).to_contain_text("TCN · 135,555 parameters")
        page.get_by_role("button", name="New run", exact=True).click()
        size = page.locator("#training-size")
        expect(size.locator("option:checked")).to_have_text("32 channels · 35,011 parameters")
        size.select_option(label="16 channels · 9,315 parameters")
        page.get_by_role("button", name="Edit JSON").click()
        editor = page.locator("#training-json")
        imported = json.loads(editor.input_value())
        assert imported["parameters"] == {"channels": 16}
        imported["parameters"].update(dropout=0.24, dilations=[1, 4], side_slopes=True)
        editor.fill(json.dumps(imported))
        page.get_by_role("button", name="Use form").click()
        expect(size.locator("option:checked")).to_have_text("Custom size (JSON)")
        size.select_option(label="64 channels · 135,555 parameters")
        page.get_by_role("button", name="Edit JSON").click()
        assert json.loads(editor.input_value())["parameters"] == {"channels": 64, "dropout": 0.24}
        page.get_by_role("button", name="Use form").click()
        page.locator("#training-architecture").select_option("baseline")
        expect(size).to_be_hidden()
        page.locator("#training-architecture").select_option("tcn")
        expect(size.locator("option:checked")).to_have_text("64 channels · 135,555 parameters")
        size.select_option(label="Custom size (JSON)")
        configuration = page.locator("#training-parameters")
        expect(configuration).to_be_focused()
        configuration.fill('{"dropout":0.0,"smoothing":0.10000000000000001}')
        size.select_option(label="48 channels · 77,091 parameters")
        updated = configuration.input_value()
        assert '"dropout": 0.0' in updated and '"smoothing": 0.10000000000000001' in updated
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    finally:
        page.close()
