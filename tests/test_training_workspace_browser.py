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
from regimex.model_registry import (
    FixedUpdatePolicy,
    adaptive_descriptor,
    baseline_descriptor,
    segment_moe_descriptor,
    tcn_descriptor,
)

ASSETS = Path(regimex.__file__).parent / "web"
ORIGIN = "https://regimex.test"
RUN_ID = str(UUID(int=1))
EVALUATION_ID = str(UUID(int=2))
PUBLICATION_ID = str(UUID(int=3))
CANDIDATE = "desk/../tcn+v1"
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
        self.archive_requests: list[dict[str, Any]] = []
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
        if url.path.split("/", 2)[1] in {"", "training", "connections"}:
            route.fulfill(path=ASSETS / "index.html", content_type="text/html")
            return
        if url.path.startswith("/assets/"):
            asset = ASSETS / Path(url.path).name
            route.fulfill(path=asset, content_type=mimetypes.guess_type(asset)[0])
            return
        path = url.path.removeprefix("/api/v1/")
        self.pages.append(path)
        if route.request.method == "POST":
            self.post(route, path)
            return
        if path == "training-runs/archive" and route.request.method == "PATCH":
            assert route.request.headers["x-csrf-token"] == "test"
            body = route.request.post_data_json
            self.archive_requests.append(body)
            for row in self.runs:
                if row["run_id"] in body["run_ids"]:
                    assert row["job"]["state"] in {"succeeded", "failed", "cancelled"}
                    row["archived_at"] = "2026-02-01T12:00:00Z" if body["archived"] else None
            route.fulfill(status=204)
            return
        if path == "training-runs":
            params = parse_qs(url.query)
            start = int(params["offset"][0])
            limit = int(params["limit"][0])
            view = params["archive"][0]
            rows = [
                row
                for row in self.runs
                if view == "all" or bool(row.get("archived_at")) == (view == "archived")
            ]
            route.fulfill(json={"items": rows[start : start + limit]})
            return
        if path.startswith("training-runs/") and path.count("/") == 1:
            run_id = path.split("/")[1]
            row = (
                self.run
                if run_id == RUN_ID
                else next(r for r in self.runs if r["run_id"] == run_id)
            )
            route.fulfill(json=row)
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
            "benchmarks": {
                "reservation": {"selection_started_at": "2026-01-01", "opened_at": None},
                "frozen_choice": None,
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
                    segment_moe_descriptor().describe().model_dump(mode="json"),
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
                self.run["job"]["launch_source"] = {"channel": "web"}
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
            assert path == "models/publish"
            assert parse_qs(urlsplit(request.url).query) == {"model_version_id": [CANDIDATE]}
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
        expect(page.get_by_text("No current runs.")).to_be_visible()
        page.get_by_role("button", name="New run", exact=True).press("Enter")
        expect(page.get_by_label("Training labels")).to_have_value("training-labels")
        expect(page.get_by_label("Training labels").locator("option")).to_have_count(1)
        expect(page.get_by_label("Validation labels")).to_have_value("validation-labels")
        expect(page.get_by_label("Validation labels").locator("option")).to_have_count(1)
        expect(page.get_by_label("Development dataset").locator("option")).to_have_count(1)
        expect(page.get_by_label("Architecture").locator("option")).to_have_text(
            [
                "Temporal convolutional network (TCN)",
                "Logistic baseline",
                "Segment MoE",
                "worker-verified",
            ]
        )
        expect(page.get_by_role("option", name="Segment MoE")).to_have_attribute(
            "value", "segment-moe"
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
        page.get_by_text("Run details and settings").click()
        expect(page.get_by_text("Started via", exact=True).locator("+ dd")).to_have_text(
            "Web interface"
        )
        page.get_by_text("Run details and settings").click()
        assert api.posts[0] == api.posts[1]
        assert api.posts[0][:2] == ("training-runs", expected_intake)

        expect(page.locator("#training-score")).to_contain_text("0.72")
        expect(page.get_by_role("link", name="Download metric data")).to_have_attribute(
            "href", f"/api/v1/training-runs/{RUN_ID}/metrics"
        )
        seed = page.get_by_label("Training repeat")
        expect(seed).to_have_value("23")
        expect(seed.locator("option:checked")).to_have_text("Seed 23 (used for this model)")
        loss = page.get_by_role("img", name="Training loss by epoch")
        expect(loss.locator("circle title")).to_have_text("Epoch 1: 0.56")
        seed.select_option("11")
        expect(seed.locator("option:checked")).to_have_text("Seed 11")
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
            "models/publish",
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
        page.goto(ORIGIN + "/training/new")
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
        page.goto(ORIGIN + "/training/new")
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
        page.goto(ORIGIN + "/training/new")
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
        page.goto(ORIGIN + "/training/new")
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


def test_workspace_limits_follow_the_selected_training_budget(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    architecture = {
        **TCN,
        "architecture_id": "custom-optimizer",
        "default_training_policy": {
            **POLICY,
            "optimizer": "lbfgs",
            "epochs": 55,
            "seeds": [5, 8, 13],
        },
        "supported_optimizers": ["lbfgs", "adamw"],
        "max_training_seeds": 2,
        "max_training_epochs": 7,
        "max_training_updates": 3,
    }
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    page.route(
        ORIGIN + "/api/v1/architectures*", lambda r: r.fulfill(json={"items": [architecture]})
    )
    try:
        page.goto(ORIGIN + "/training/new")
        page.get_by_text("Training policy", exact=True).click()
        epochs = page.get_by_label("Maximum epochs")
        repeats = page.get_by_label("Training repeats")
        seeds = page.get_by_label("Random seeds", exact=True)
        expect(epochs).to_have_value("7")
        expect(epochs).to_have_attribute("max", "7")
        expect(repeats).to_have_attribute("max", "2")
        expect(seeds).to_have_value("5, 8")
        expect(page.get_by_label("Optimizer", exact=True)).to_have_value("lbfgs")
        start = page.get_by_role("button", name="Start training")
        epochs.fill("8")
        start.click()
        expect(epochs).to_be_focused()
        expect(epochs).to_have_value("8")
        epochs.fill("6")
        page.get_by_role("button", name="Edit JSON").click()
        editor = page.get_by_label("Run configuration (JSON)")
        use_form = page.get_by_role("button", name="Use form")
        # Omitting epochs means the API default of 30, not the form's suggested 7.
        editor.fill('{"optimizer":"lbfgs","seeds":[5]}')
        use_form.click()
        expect(page.locator("#training-prerequisite")).to_contain_text("7 epochs per repeat")
        start.click()
        expect(page.locator("#training-notice")).to_contain_text("7 epochs per repeat")
        # Raw and form submissions enforce the workspace's update count per repeat.
        fixed = {"updates": 4, "validation_updates": [4], "seeds": [5, 8]}
        editor.fill(json.dumps(fixed))
        start.click()
        expect(page.locator("#training-notice")).to_contain_text("3 updates per repeat")
        expect(editor).to_have_value(json.dumps(fixed))
        assert not api.posts
        fixed.update(updates=3, validation_updates=[3])
        editor.fill(json.dumps(fixed))
        use_form.click()
        expect(editor).to_be_hidden()
        updates = page.get_by_label("Updates per seed")
        expect(updates).to_have_attribute("max", "3")
        expect(updates).to_have_value("3")
        expect(seeds).to_have_value("5, 8")
        page.get_by_role("button", name="Help for Updates per seed").click()
        expect(page.get_by_role("tooltip")).to_contain_text("3 updates per repeat")
        page.keyboard.press("Escape")
        start.click()
        expect(page.locator("#training-detail")).to_be_visible()
        assert api.posts[0][1]["policy"] == fixed
    finally:
        page.close()


def test_one_repeat_workspace_keeps_tcn_epoch_controls(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    architecture = tcn_descriptor().describe(max_training_seeds=1).model_dump(mode="json")
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    page.route(
        ORIGIN + "/api/v1/architectures*", lambda r: r.fulfill(json={"items": [architecture]})
    )
    try:
        page.goto(ORIGIN + "/training/new")
        page.get_by_text("Training policy", exact=True).click()
        expect(page.get_by_label("Training repeats")).to_have_attribute("max", "1")
        expect(page.get_by_label("Random seeds", exact=True)).to_have_value("11")
        page.get_by_label("Maximum epochs").fill("2")
        page.get_by_role("button", name="Start training").click()
        expect(page.locator("#training-detail")).to_be_visible()
        assert api.posts[0][1]["policy"] == {
            **architecture["default_training_policy"],
            "seeds": [11],
            "epochs": 2,
        }
    finally:
        page.close()


def test_json_policy_requires_integer_tokens_in_both_editors(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(ORIGIN + "/training/new")
        page.get_by_role("button", name="Edit JSON").click()
        editor = page.get_by_label("Run configuration (JSON)")
        use_form = page.get_by_role("button", name="Use form")
        start = page.get_by_role("button", name="Start training")
        for field, literal in (
            ("epochs", "2.0"),
            ("seeds", "[11e0]"),
            ("validation_updates", "[3.0]"),
        ):
            editor.fill('{"optimizer":"adamw","' + field + '":' + literal + "}")
            use_form.click()
            expect(editor).to_be_visible()
            expect(page.locator("#training-prerequisite")).to_contain_text(
                "JSON integer for " + field
            )
            start.click()
            expect(page.locator("#training-notice")).to_contain_text("JSON integer for " + field)
        assert not api.posts
        editor.fill('{"optimizer":"adamw","epochs":2,"weight_decay":2.0}')
        use_form.click()
        expect(editor).to_be_hidden()
        expect(page.get_by_label("Maximum epochs")).to_have_value("2")
        start.click()
        expect(page.locator("#training-detail")).to_be_visible()
        assert api.posts[0][1]["policy"]["epochs"] == 2
    finally:
        page.close()


def test_repeat_count_and_generated_seeds_round_trip_through_run_json(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    page = browser.new_page(viewport={"width": 320, "height": 640})
    page.route(ORIGIN + "/**", api.dispatch)
    architecture = {**TCN, "default_training_policy": POLICY, "max_training_seeds": 5}
    page.route(
        ORIGIN + "/api/v1/architectures*", lambda r: r.fulfill(json={"items": [architecture]})
    )
    try:
        page.goto(ORIGIN + "/training/new")
        page.get_by_text("Training policy", exact=True).click()
        repeats = page.get_by_label("Training repeats", exact=True)
        seeds = page.get_by_label("Random seeds", exact=True)
        expect(repeats).to_have_value(str(len(POLICY["seeds"])))
        seeds.fill("0, 4294967295, 7")
        expect(repeats).to_have_value("3")
        repeats.fill("1")
        expect(seeds).to_have_value("0")
        # Each draw repeats once so extending a list must handle collisions.
        page.evaluate("""() => {
            let draws = 0;
            crypto.getRandomValues = words => {
                for (let i = 0; i < words.length; i++) words[i] = Math.floor(draws++ / 2);
                return words;
            };
        }""")
        expect(repeats).to_have_attribute("max", "5")
        repeats.fill("5")
        expanded = [int(value) for value in seeds.input_value().split(",")]
        assert len(expanded) == len(set(expanded)) == 5 and expanded[0] == 0
        repeats.fill("6")
        start = page.get_by_role("button", name="Start training")
        start.click()
        expect(repeats).to_be_focused()
        assert not api.posts

        repeats.fill("4")
        generate = page.get_by_role("button", name="Generate seeds", exact=True)
        for control in (repeats, seeds, generate):
            box = control.bounding_box()
            assert box and 0 <= box["x"] < box["x"] + box["width"] <= 320
        generate.press("Enter")
        generated = [int(value) for value in seeds.input_value().split(",")]
        assert len(generated) == len(set(generated)) == 4 and generated != expanded[:4]
        assert all(0 <= value < 2**32 for value in generated)
        page.get_by_role("button", name="Edit JSON").click()
        expected_policy = {**POLICY, "seeds": generated}
        editor = page.get_by_label("Run configuration (JSON)")
        assert json.loads(editor.input_value())["policy"] == expected_policy
        page.get_by_role("button", name="Use form").click()
        expect(repeats).to_have_value("4")
        start.click()
        expect(page.get_by_role("heading", name="TCN · development")).to_be_visible()
        assert api.posts[0][1]["policy"] == expected_policy
    finally:
        page.close()


@pytest.mark.parametrize("custom", [False, True], ids=["automatic", "custom"])
def test_update_budget_submits_only_update_settings(browser: Any, custom: bool) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(ORIGIN + "/training/new")
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
        page.goto(ORIGIN + "/training/new")
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
            expect(page.get_by_label("Training repeats", exact=True)).to_have_value("1")
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
        page.goto(ORIGIN + "/training/new")
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
        expect(editor).to_be_hidden()
        assert (
            json.loads(page.locator("#training-runtime").input_value())
            == request["training_runtime"]
        )
        start.click()
        expect(page.get_by_role("heading", name="Price sensitivity")).to_be_visible()
        assert api.posts[0][1] == request
    finally:
        page.close()


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ('{"trees":999}', "Number of trees must be a whole number from 1 to 128."),
        ('{"trees":30.00000000000000001}', "Number of trees"),
        ('{"feature_set":"unavailable"}', "Price features"),
    ],
)
def test_discard_after_invalid_json_import_restores_original_draft(
    browser: Any, parameters: str, message: str
) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    forest = adaptive_descriptor().describe().model_dump(mode="json")
    page.route(
        ORIGIN + "/api/v1/architectures*", lambda r: r.fulfill(json={"items": [TCN, forest]})
    )
    try:
        page.goto(ORIGIN + "/training/new")
        architecture = page.get_by_label("Architecture")
        trees = page.get_by_label("Number of trees", exact=True)
        architecture.select_option("adaptive")
        trees.fill("24")
        architecture.select_option("tcn")
        edit = page.get_by_role("button", name="Edit JSON")
        edit.click()
        editor = page.get_by_label("Run configuration (JSON)")
        original = {
            **json.loads(editor.input_value()),
            "display_name": "Original TCN",
            "policy": {"optimizer": "adamw", "seeds": [17, 31], "epochs": 7},
            "training_runtime": {"memory_budget_bytes": 9007199254740993},
        }
        text = json.dumps(original, indent=1)
        editor.fill(text)
        use_form = page.get_by_role("button", name="Use form")
        use_form.click()
        expect(editor).to_be_hidden()
        edit.click()
        expect(editor).to_have_value(text)
        pasted = json.dumps(
            {
                **original,
                "architecture_id": "adaptive",
                "display_name": "Changed forest",
                "policy": {"optimizer": "extra_trees", "seeds": [99]},
                "parameters": {},
            }
        ).replace('"parameters": {}', '"parameters": ' + parameters)
        editor.fill(pasted)
        use_form.click()
        expect(editor).to_be_focused()
        expect(editor).to_have_value(pasted)
        expect(editor).to_have_attribute("aria-invalid", "true")
        expect(editor).to_have_attribute("aria-errormessage", "training-prerequisite")
        expect(page.locator("#training-prerequisite")).to_contain_text(message)
        page.get_by_role("button", name="Discard JSON edits").click()
        expect(architecture).to_have_value("tcn")
        expect(page.get_by_label("Run name (optional)")).to_have_value("Original TCN")
        expect(page.get_by_label("Random seeds", exact=True)).to_have_value("17, 31")
        architecture.select_option("adaptive")
        expect(trees).to_have_value("24")
        architecture.select_option("tcn")
        edit.click()
        expect(editor).to_have_value(text)
        use_form.click()
        page.get_by_role("button", name="Start training").click()
        expect(page.locator("#training-detail")).to_be_visible()
        assert api.posts[0][1] == original
    finally:
        page.close()


@pytest.mark.parametrize("show_legacy_descriptor", [False, True], ids=["current", "mixed"])
def test_old_segment_moe_run_opens_and_copies_under_its_current_architecture(
    browser: Any, show_legacy_descriptor: bool
) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    descriptor = segment_moe_descriptor().describe().model_dump(mode="json")
    available = [descriptor]
    if show_legacy_descriptor:
        available.insert(0, {**descriptor, "architecture_id": "persistence"})
    api.run["training"].update(
        architecture_id="persistence",
        policy=descriptor["default_training_policy"],
        training_runtime={"memory_budget_bytes": 9007199254740993},
    )
    api.runs.append(api.run)
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    page.route(ORIGIN + "/api/v1/architectures*", lambda r: r.fulfill(json={"items": available}))
    try:
        page.goto(f"{ORIGIN}/training/{RUN_ID}")
        expect(page.get_by_role("heading", name="Segment MoE · development")).to_be_visible()
        saved = json.loads(page.locator("#training-settings").text_content())
        assert saved["architecture_id"] == "persistence"
        page.get_by_role("button", name="New run from this run").click()
        picker = page.get_by_label("Architecture")
        expect(picker).to_have_value("segment-moe")
        expect(picker.locator("option")).to_have_text(["Segment MoE"])
        page.get_by_role("button", name="Edit JSON").click()
        editor = page.get_by_label("Run configuration (JSON)")
        assert json.loads(editor.input_value()) == {**saved, "architecture_id": "segment-moe"}
        editor.fill(json.dumps(saved))
        page.get_by_role("button", name="Start training").click()
        expect(page.locator("#training-detail")).to_be_visible()
        assert api.posts[0][1] == {**saved, "architecture_id": "segment-moe"}
    finally:
        page.close()


def test_copy_opens_unrepresentable_settings_in_json_with_a_working_form(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    forest = adaptive_descriptor().describe().model_dump(mode="json")
    api.run["training"].update(
        architecture_id="adaptive",
        parameters={"trees": 999},
        policy=forest["default_training_policy"],
    )
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    page.route(
        ORIGIN + "/api/v1/architectures*", lambda r: r.fulfill(json={"items": [TCN, forest]})
    )
    try:
        page.goto(f"{ORIGIN}/training/{RUN_ID}")
        page.get_by_role("button", name="New run from this run").click()
        editor = page.get_by_label("Run configuration (JSON)")
        expect(editor).to_be_focused()
        expect(page.locator("#training-prerequisite")).to_contain_text("Number of trees")
        assert json.loads(editor.input_value())["parameters"] == {"trees": 999}
        page.get_by_role("button", name="Discard JSON edits").click()
        expect(page.get_by_label("Architecture")).to_have_value("tcn")
        expect(page.get_by_role("button", name="Start training")).to_be_enabled()
        assert not api.posts
    finally:
        page.close()


@pytest.mark.parametrize("edit_json", [False, True], ids=["form", "policy-paste"])
def test_advanced_json_preserves_numeric_literals(browser: Any, edit_json: bool) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(ORIGIN + "/training/new")
        page.locator("#training-advanced-settings > summary").click()
        page.locator("#training-parameters").fill('{"plugin":[8.0,"8.0",1e-7,-0.0]}')
        page.locator("#training-runtime").fill('{"memory_budget_bytes":9007199254740993}')
        page.locator("#training-inference").fill('{"model_cache_budget_bytes":9007199254740995}')
        if edit_json:
            page.get_by_role("button", name="Edit JSON").click()
            editor = page.get_by_label("Run configuration (JSON)")
            assert "9007199254740993" in editor.input_value()
            editor.fill(json.dumps(POLICY))
            page.get_by_role("button", name="Use form").click()
            expect(editor).to_be_hidden()
            assert "8.0" in page.locator("#training-parameters").input_value()
            page.get_by_label("Run name (optional)").fill("Edited after import")
        with page.expect_request(
            lambda r: r.method == "POST" and r.url.endswith("/training-runs")
        ) as request:
            page.get_by_role("button", name="Start training").click()
        raw = request.value.post_data
        assert raw is not None
        assert all(
            token in raw
            for token in ("8.0", "1e-7", "-0.0", "9007199254740993", "9007199254740995")
        )
        parameters = json.loads(raw)["parameters"]["plugin"]
        assert isinstance(parameters[0], float) and parameters[1] == "8.0"
        expect(page.locator("#training-detail")).to_be_visible()
    finally:
        page.close()


@pytest.mark.parametrize(
    "return_before_ack", [False, True], ids=["submitted-draft", "edited-draft"]
)
def test_training_acknowledgement_after_navigation(browser: Any, return_before_ack: bool) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    held: list[Any] = []
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    page.route(ORIGIN + "/api/v1/training-runs", lambda route: held.append(route))
    try:
        url = ORIGIN + "/training/new"
        page.goto(url)
        name = page.get_by_label("Run name (optional)")
        name.fill("Submitted draft")
        with page.expect_request(lambda r: r.method == "POST"):
            page.get_by_role("button", name="Start training").click()
        page.locator("#open-analyses").click()
        expect(page).to_have_url(ORIGIN + "/")
        if return_before_ack:
            page.go_back()
            expect(name).to_have_value("Submitted draft")
            name.fill("Next draft")
        with page.expect_response(lambda r: r.request.method == "POST"):
            api.post(held.pop(), "training-runs")
        assert len(api.posts) == 1
        if return_before_ack:
            expect(page.locator("#training-runs")).to_contain_text("Submitted draft")
            expect(page).to_have_url(url)
            expect(name).to_have_value("Next draft")
            # Reverting that edit still refers to the accepted submission.
            name.fill("Submitted draft")
            with page.expect_request(lambda r: r.method == "POST"):
                page.get_by_role("button", name="Start training").click()
            api.post(held.pop(), "training-runs")
            assert api.posts[0] == api.posts[1]
            expect(page).to_have_url(ORIGIN + "/training/" + RUN_ID)
        else:
            page.go_back()
            expect(page).to_have_url(ORIGIN + "/training/" + RUN_ID)
            expect(page.get_by_role("heading", name="Submitted draft")).to_be_visible()
            page.get_by_role("button", name="New run", exact=True).click()
            expect(name).to_have_value("")
    finally:
        page.close()


def test_explicit_new_run_can_use_the_same_settings_as_a_pending_submission(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    held: list[Any] = []
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    page.route(ORIGIN + "/api/v1/training-runs", lambda route: held.append(route))
    try:
        url = ORIGIN + "/training/new"
        page.goto(url)
        start = page.get_by_role("button", name="Start training")
        with page.expect_request(lambda r: r.method == "POST"):
            start.click()
        page.locator("#open-analyses").click()
        page.locator("#open-training").click()
        page.get_by_role("button", name="New run", exact=True).click()
        with page.expect_request(lambda r: r.method == "POST"):
            start.click()
        first, second = held
        assert first.request.post_data == second.request.post_data
        assert first.request.headers["idempotency-key"] != second.request.headers["idempotency-key"]
        with page.expect_response(lambda r: r.request.method == "POST"):
            api.post(first, "training-runs")
        expect(page.locator("#training-runs li")).to_have_count(1)
        expect(page).to_have_url(url)
        expect(start).to_be_disabled()
        api.post(second, "training-runs")
        expect(page).to_have_url(ORIGIN + "/training/" + RUN_ID)
    finally:
        page.close()


def test_invalid_json_draft_survives_back_forward_until_discarded(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(ORIGIN + "/training")
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
        expect(page.get_by_text("No current runs.")).to_be_visible()
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
    rows = page.locator("#training-runs button[data-run-id]")
    try:
        page.goto(ORIGIN + "/training/new")
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
        expect(page.get_by_role("combobox", name="Run history view")).to_be_focused()
        page.keyboard.press("Tab")
        expect(page.get_by_role("button", name="Select runs")).to_be_focused()
        page.keyboard.press("Tab")
        expect(rows.first).to_be_focused()
        page.get_by_role("button", name="Older").press("Enter")
        expect(rows).to_have_count(1)
        rows.first.press("Enter")
        heading = page.get_by_role("heading", name="TCN · development")
        expect(heading).to_be_focused()
        expect(toggle).to_have_attribute("aria-expanded", "false")
        expect(page.locator("#training-history")).to_be_hidden()
        expect(page).to_have_url(ORIGIN + "/training/" + api.runs[-1]["run_id"])
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


@pytest.mark.parametrize("open_list", [True, False], ids=["run-list", "run-detail"])
def test_delayed_run_load_preserves_the_keyboard_position(browser: Any, open_list: bool) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    pending: list[Any] = []
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    path = "/training" if open_list else f"/training/{RUN_ID}"
    request_url = f"{ORIGIN}/api/v1/training-runs" + ("?*" if open_list else f"/{RUN_ID}")
    page.route(request_url, lambda route: pending.append(route))
    try:
        with page.expect_request(request_url):
            page.goto(ORIGIN + path)
        new = page.get_by_role("button", name="New run", exact=True)
        new.focus()
        if open_list:
            pending[0].fulfill(json={"items": [api.run]})
            expect(page.locator("#training-runs [data-run-id]")).to_have_count(1)
        else:
            pending[0].fulfill(json=api.run)
            expect(page.locator("#training-detail-title")).to_be_visible()
        expect(new).to_be_focused()
        expect(page).to_have_url(ORIGIN + path)
    finally:
        page.close()


@pytest.mark.parametrize("path", ["/", f"/training/{RUN_ID}?view=all", f"/connections/{RUN_ID}"])
def test_analysis_user_has_no_administrative_controls(browser: Any, path: str) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI(role="user")
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(ORIGIN + path)
        expect(page.locator("#workspace")).to_be_visible()
        expect(page).to_have_url(ORIGIN + "/")
        expect(page.get_by_role("link", name="Training", exact=True)).to_be_hidden()
        expect(page.get_by_role("link", name="Connections", exact=True)).to_be_hidden()
        assert not any(p.startswith(("training-runs", "admin/integrations")) for p in api.pages)
    finally:
        page.close()


def test_training_bookmark_restores_session_and_defers_analysis_requests(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    page.route(
        ORIGIN + "/api/v1/session",
        lambda route: route.fulfill(status=503, json={"error": {"message": "Restarting"}}),
        times=1,
    )
    try:
        page.goto(ORIGIN + "/training")
        expect(page.get_by_text("Reconnecting to the service…")).to_be_visible()
        expect(page.get_by_role("heading", name="Sign in")).to_be_hidden()
        expect(page.get_by_text("No current runs.")).to_be_visible()
        analysis_requests = {"storage", "deletions", "experiments", "models"}
        assert analysis_requests.isdisjoint(api.pages)
        page.get_by_role("link", name="Analyses").click()
        expect(page.locator("#history-empty")).to_be_visible()
        expect(page.locator("#storage-usage")).to_be_visible()
        assert analysis_requests.issubset(api.pages)
        page.go_back()
        expect(page.get_by_text("No current runs.")).to_be_visible()
        page.go_forward()
        expect(page.locator("#history-empty")).to_be_visible()
        assert api.pages.count("experiments") == 1
    finally:
        page.close()


@pytest.mark.parametrize("leave", [False, True], ids=["reconnect", "navigate-away"])
def test_training_bookmark_survives_an_interrupted_first_read(browser: Any, leave: bool) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    page = browser.new_page()
    page.clock.install()
    page.route(ORIGIN + "/**", api.dispatch)
    page.route(
        f"{ORIGIN}/api/v1/training-runs/{RUN_ID}",
        lambda route: route.fulfill(status=503, json={"error": {"message": "Restarting"}}),
        times=1,
    )
    try:
        page.goto(f"{ORIGIN}/training/{RUN_ID}")
        expect(page.locator("#training-sync")).to_be_visible()
        if leave:
            page.get_by_role("button", name="New run", exact=True).click()
            expect(page).to_have_url(f"{ORIGIN}/training/new")
        page.clock.fast_forward(3000)
        if leave:
            assert f"training-runs/{RUN_ID}" not in api.pages
        else:
            expect(page.get_by_role("link", name="Download metric data")).to_be_visible()
            assert api.pages.count(f"training-runs/{RUN_ID}") == 1
        expect(page.locator("#training-sync")).to_be_hidden()
        assert not api.posts
    finally:
        page.close()


def test_training_metrics_retry_an_interrupted_read(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    page.route(
        ORIGIN + f"/api/v1/training-runs/{RUN_ID}/metrics",
        lambda route: route.fulfill(status=408, json={"error": {"message": "Request timeout"}}),
        times=1,
    )
    try:
        page.goto(ORIGIN + f"/training/{RUN_ID}")
        expect(page.locator("#training-sync")).to_be_visible()
        expect(page.get_by_role("link", name="Download metric data")).to_be_visible()
        expect(page.locator("#training-sync")).to_be_hidden()
        assert not api.posts
    finally:
        page.close()


def test_training_fragment_bookmark_replaces_its_history_entry(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        previous = ORIGIN + "/training/new"
        canonical = f"{ORIGIN}/training/{RUN_ID}?view=archived"
        page.goto(previous)
        expect(page.get_by_role("heading", name="New training run")).to_be_visible()
        length = page.evaluate("history.length")
        page.goto(f"{ORIGIN}/#training/{RUN_ID}?view=archived")
        expect(page).to_have_url(canonical)
        expect(page.get_by_role("combobox", name="Run history view")).to_have_value("archived")
        expect(page.locator("#training-state")).to_have_text("Complete")
        assert page.evaluate("history.length") == length + 1
        page.go_back()
        expect(page).to_have_url(previous)
        page.go_forward()
        expect(page).to_have_url(canonical)
    finally:
        page.close()


def test_connection_bookmark_reloads_and_preserves_browser_history(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    connection_id = str(UUID(int=81))
    connection = {
        "integration_id": connection_id,
        "display_name": "Market simulator",
        "source_namespace": "market",
        "owner_id": "desk",
        "enabled": True,
        "revision": 0,
    }
    api_path = "/api/v1/admin/integrations"
    detail = f"{api_path}/{connection_id}"
    responses = {
        api_path: {"items": [connection]},
        detail: connection,
        f"{detail}/keys": {"items": []},
        f"{detail}/events": {"items": []},
        f"{detail}/storage": {"used_bytes": 0, "reserved_bytes": 0, "limit_bytes": 1024},
    }
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    page.route(
        ORIGIN + api_path + "**",
        lambda route: route.fulfill(json=responses[urlsplit(route.request.url).path]),
    )
    try:
        catalog = ORIGIN + "/connections"
        bookmark = f"{catalog}/{connection_id}"
        page.goto(catalog)
        expect(page.get_by_role("link", name="Connections")).to_have_attribute(
            "href", "/connections"
        )
        page.get_by_role("link", name="Market simulator").click()
        heading = page.get_by_role("heading", name="Market simulator")
        expect(page).to_have_url(bookmark)
        expect(heading).to_be_visible()
        page.reload()
        expect(heading).to_be_visible()
        page.go_back()
        expect(page).to_have_url(catalog)
        expect(heading).to_be_hidden()
        page.go_forward()
        expect(page).to_have_url(bookmark)
        expect(heading).to_be_visible()
    finally:
        page.close()


def test_history_preview_exposes_full_metadata_on_hover_and_keyboard_focus(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    dataset = "dataset-12345678abcdef0012345678abcdef00"
    api.run.update(display_name="Morning session", trainable_parameters=135555)
    api.run["training"]["dataset_id"] = dataset
    unnamed = completed_run(str(UUID(int=99)), "training")
    unnamed.update(trainable_parameters=9315)
    api.runs = [api.run, unnamed]
    page = browser.new_page(viewport={"width": 1280, "height": 900}, locale="en-US")
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        new_url = ORIGIN + "/training/new"
        page.goto(new_url)
        named = page.locator(f'button[data-run-id="{RUN_ID}"]')
        fallback = page.locator(f'button[data-run-id="{unnamed["run_id"]}"]')
        expect(named.locator("strong")).to_have_text("Morning session")
        expect(named.locator("time")).to_be_visible()
        expect(named.locator("[data-state]")).to_have_text("Complete")
        expect(named).not_to_contain_text("parameters")
        expect(named).not_to_contain_text(dataset)
        expect(fallback.locator("strong")).to_have_text("TCN · development")
        expect(fallback).not_to_contain_text("9,315")

        note = page.get_by_role("tooltip")
        named.hover()
        expect(note).to_be_visible()
        expect(note).to_contain_text("Temporal convolutional network")
        expect(note).to_contain_text(dataset)
        expect(note).to_contain_text("135,555")
        page.clock.install()
        note.hover()
        page.keyboard.press("Escape")
        # Dismissing the overlay can uncover another row beneath a stationary pointer.
        page.clock.run_for(600)
        expect(note).to_be_hidden()
        fallback.hover(position={"x": 8, "y": 8})
        page.clock.run_for(600)
        expect(note).to_contain_text("9,315")

        select = page.get_by_role("button", name="Select runs", exact=True)
        select.hover()
        select.focus()
        page.keyboard.press("Tab")
        expect(named).to_be_focused()
        expect(note).to_be_visible()
        expect(named).to_have_attribute("aria-describedby", str(note.get_attribute("id")))
        page.keyboard.press("Escape")
        expect(note).to_be_hidden()
        expect(named).to_be_focused()
        expect(page).to_have_url(new_url)
        assert f"training-runs/{RUN_ID}" not in api.pages
    finally:
        page.close()


def test_touch_history_preview_and_bulk_selection_leave_first_tap_navigation_intact(
    browser: Any,
) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.run["display_name"] = "Evening session"
    api.runs.append(api.run)
    page = browser.new_page(viewport={"width": 390, "height": 800}, has_touch=True)
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        new_url = ORIGIN + "/training/new"
        page.goto(new_url)
        page.get_by_role("button", name="Run history", exact=True).tap()
        page.get_by_role("button", name="Select runs", exact=True).tap()
        checkbox = page.get_by_role("checkbox", name="Select Evening session", exact=True)
        checkbox.tap()
        expect(checkbox).to_be_checked()

        note = page.get_by_role("tooltip")
        expect(note).to_be_hidden()
        details = page.get_by_role("button", name="Details for Evening session", exact=True)
        details.tap()
        expect(note).to_be_visible()
        expect(note).to_contain_text("development")
        expect(note).not_to_contain_text("parameters")
        expect(page).to_have_url(new_url)
        page.get_by_text("1 selected", exact=True).tap()
        expect(note).to_be_hidden()
        expect(checkbox).to_be_checked()
        assert not api.archive_requests

        page.locator(f'button[data-run-id="{RUN_ID}"]').tap()
        expect(page.get_by_role("heading", name="Evening session")).to_be_focused()
        expect(page.locator("#training-history")).to_be_hidden()
        expect(page).to_have_url(f"{ORIGIN}/training/{RUN_ID}")
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
        page.goto(f"{ORIGIN}/training/{RUN_ID}")
        title = page.get_by_role("heading", name=name)
        expect(title).to_be_visible()
        assert title.evaluate("el => el.clientWidth >= el.scrollWidth")
        page.get_by_role("button", name="Run history").click()
        row = page.locator("#training-runs button[data-run-id]")
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
        page.goto(f"{ORIGIN}/training/{RUN_ID}")
        heading = page.locator("#training-detail-title")
        row = page.locator("#training-runs button[data-run-id][aria-current=true]")
        expect(heading).to_have_text(default_title)
        expect(row.locator("strong")).to_have_text(default_title)
        page.get_by_text("Run details and settings").click()
        expect(page.locator("#training-facts")).to_contain_text(dataset_id)
        expect(page.get_by_text("Started via", exact=True).locator("+ dd")).to_have_text(
            "Not recorded"
        )

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
        expect(row.locator("strong")).to_have_text("Opening Δ")
        expect(page.get_by_role("button", name="Details for Opening Δ")).to_be_visible()
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
        expect(row.locator("strong")).to_have_text(default_title)
        expect(page.get_by_role("button", name=f"Details for {default_title}")).to_be_visible()
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
            expect(rows.locator("button[data-run-id]")).to_have_attribute(
                "data-run-id", api.runs[-1]["run_id"]
            )
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
            expect(page.locator("#training-downstream")).to_contain_text("Running")
            expect(page.locator("#training-evaluate")).to_be_hidden()
            page.get_by_role("button", name="Refresh", exact=True).click()
            if kind == "evaluation":
                expect(publish).to_be_enabled()
            else:
                expect(page.get_by_text("New analyses use this model by default.")).to_be_visible()
        assert [path for path, _, _ in api.posts] == ["evaluations", "models/publish"]
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
            page.goto(ORIGIN + "/training/" + RUN_ID)
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
    attempt["job"]["launch_source"] = {"channel": "api"}
    if state == "failed":
        attempt["job"]["failure"] = {"message": "Worker timed out."}
    api.actions.append(attempt)
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(f"{ORIGIN}/training/{RUN_ID}")
        actions = page.locator("#training-actions")
        expect(actions).to_contain_text(state.capitalize())
        expect(actions).to_contain_text("Launch source: API")
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
        page.goto(f"{ORIGIN}/training/{RUN_ID}")
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


@pytest.mark.parametrize(
    ("experiment_count", "bounds", "point_defined", "interval_text"),
    [
        (3, None, True, None),
        (1, (None, None), True, None),
        (1, ([0.6, 0.6], [0.8, 0.8]), True, None),
        (3, ([0.7, 0.5], None), True, None),
        (3, ([0.5, 0.7], [0.0, 0.0]), False, "95% CI · 0.5\u20130.7"),
        (3, ([0.0, 0.0], None), True, "95% CI · 0\u20130"),
    ],
    ids=["no-bootstrap", "null-bounds", "one-experiment", "reversed", "no-labeled-steps", "zero"],
)
def test_evaluation_only_displays_intervals_for_defined_scores(
    browser: Any,
    experiment_count: int,
    bounds: tuple[list[float] | None, list[float] | None] | None,
    point_defined: bool,
    interval_text: str | None,
) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.actions.append(completed_run(EVALUATION_ID, "evaluation"))
    report = evaluation_report()
    report["experiment_count"] = experiment_count
    report["point_metrics"]["defined"] = point_defined
    if not point_defined:
        report["point_macro_f1"] = 0.0
    if bounds is not None:
        if bounds[0] == [0, 0]:
            report["segment_macro_f1"] = 0
        report["confidence_intervals"] = {
            "confidence_level": 0.95,
            "repetitions": 1000,
            "experiment_count": report["experiment_count"],
            "segment_macro_f1_percentile_interval": bounds[0],
            "point_macro_f1_percentile_interval": bounds[1],
        }
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    page.route(f"**/evaluations/{EVALUATION_ID}/result", lambda route: route.fulfill(json=report))
    try:
        page.goto(f"{ORIGIN}/training/{RUN_ID}")
        scores = page.locator("#training-actions .evaluation-summary")
        segment = scores.get_by_text("Interval macro-F1", exact=True).locator("+ dd")
        point = scores.get_by_text("Per-step macro-F1", exact=True).locator("+ dd")
        expect(point).to_have_text("0.8" if point_defined else "No labeled steps")
        if interval_text is None:
            expect(segment).to_have_text("0.6")
        else:
            expect(segment.locator(".evaluation-score-interval")).to_have_text(interval_text)
        expect(point.locator(".evaluation-score-interval")).to_have_count(0)
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
        page.goto(f"{ORIGIN}/training/{RUN_ID}")
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


def test_hidden_training_tab_waits_to_read_new_measurements(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.run["job"].update(state="running", stage="training", attempt=1)
    api.progress_rows = [{"epoch": 1, "training_loss": 0.8}]
    path = f"training-runs/{RUN_ID}"
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.clock.install()
        page.goto(f"{ORIGIN}/training/{RUN_ID}")
        latest = page.get_by_role("img", name="Training loss by epoch").locator("[data-last] title")
        expect(latest).to_have_text("Epoch 1: 0.8")

        page.evaluate("""() => {
            Object.defineProperty(document, 'hidden', {value: true, configurable: true});
            document.dispatchEvent(new Event('visibilitychange'));
        }""")
        before = (api.pages.count(path), api.pages.count(path + "/progress"))
        api.progress_rows.append({"epoch": 2, "training_loss": 0.6})
        page.clock.fast_forward(30_000)
        page.wait_for_load_state("networkidle")
        assert (api.pages.count(path), api.pages.count(path + "/progress")) == before

        with page.expect_response(lambda r: "/progress?attempt=1&after=1" in r.url):
            page.evaluate("""() => {
                delete document.hidden;
                document.dispatchEvent(new Event('visibilitychange'));
            }""")
            page.clock.fast_forward(100)
        expect(latest).to_have_text("Epoch 2: 0.6")
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
        page.goto(f"{ORIGIN}/training/{RUN_ID}")
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
        expect(page.locator("#training-score")).to_have_text(
            "0.72 validation interval macro-F1 · seed 23"
        )
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
        page.goto(f"{ORIGIN}/training/{RUN_ID}")
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
        expect(validation.locator("[data-last]")).to_be_visible()
        last_loss = loss.locator(".training-point[data-last]").bounding_box()
        last_validation = validation.locator(".training-point[data-last]").bounding_box()
        assert last_loss and last_validation
        loss_x = last_loss["x"] + last_loss["width"] / 2
        validation_x = last_validation["x"] + last_validation["width"] / 2
        assert loss_x == pytest.approx(validation_x, abs=1)
        curve = loss.locator("polyline").bounding_box()
        assert curve and curve["width"] > 100
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
        page.goto(f"{ORIGIN}/training/{RUN_ID}")
        heading = page.get_by_role("heading", name="Baseline · development")
        expect(heading).to_be_visible()
        chart = page.get_by_role("img", name="Validation interval macro-F1 by regularization")
        expect(chart).to_be_visible()
        expect(chart.locator(".training-point")).to_have_count(2)
        best = chart.locator(".training-point[data-best]")
        latest = chart.locator(".training-point[data-last]")
        expect(best.locator("title")).to_contain_text("0.1")
        expect(latest.locator("title")).to_contain_text("10")
        assert float(best.get_attribute("cx").removesuffix("%")) == pytest.approx(50)
        assert float(latest.get_attribute("cx").removesuffix("%")) == pytest.approx(100)
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
        page.goto(ORIGIN + "/training")
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
        seed.fill("")
        page.get_by_role("button", name="Generate seed", exact=True).click()
        assert 0 <= int(seed.input_value()) < 2**32
        expect(parameters).to_have_value("{}")
        for label in (
            "Training budget",
            "Training repeats",
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
        expect(page.get_by_text("No current runs.")).to_be_visible()
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
        expect(page.get_by_role("heading", name="Training", exact=True)).to_be_visible()
        expect(page).to_have_url(ORIGIN + "/training")
        page.reload()
        expect(page.get_by_role("heading", name="Training", exact=True)).to_be_visible()
        expect(page.get_by_role("link", name="Analyses")).to_have_js_property(
            "href", comparison_url
        )

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
        page.goto(f"{ORIGIN}/training/{RUN_ID}")
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


@pytest.mark.parametrize("width", [360, 1280], ids=["phone", "desktop"])
def test_adaptive_form_edits_forest_settings_and_round_trips_json(browser: Any, width: int) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    page = browser.new_page(viewport={"width": width, "height": 800})
    page.route(ORIGIN + "/**", api.dispatch)
    adaptive = adaptive_descriptor().describe().model_dump(mode="json")
    page.route(
        ORIGIN + "/api/v1/architectures*", lambda r: r.fulfill(json={"items": [adaptive, TCN]})
    )
    try:
        page.goto(ORIGIN + "/training")
        page.get_by_role("button", name="New run", exact=True).click()
        expect(page.get_by_label("Architecture")).to_have_value("adaptive")
        trees = page.get_by_label("Number of trees", exact=True)
        depth = page.get_by_label("Maximum tree depth", exact=True)
        features = page.get_by_label("Price features", exact=True)
        weights = page.get_by_label("Training weights", exact=True)
        expect(trees).to_have_value("40")
        expect(depth).to_have_value("12")
        expect(features).to_have_value("greedy")
        expect(weights).to_have_value("per_step")
        gap, row_gap = page.locator("#training-model-fields").evaluate(
            """grid => {
                const field = document.querySelector('#training-architecture');
                const gap = grid.getBoundingClientRect().top - field.getBoundingClientRect().bottom;
                return [gap, parseFloat(getComputedStyle(grid).rowGap)];
            }"""
        )
        assert gap == pytest.approx(row_gap) and gap > 0
        features.select_option("multiscale")
        weights.select_option("interval")
        trees.fill("24")
        depth.fill("8")
        page.get_by_text("Training policy", exact=True).click()
        expect(page.get_by_label("Maximum epochs")).to_have_count(0)
        expect(page.get_by_label("Random seed", exact=True)).to_have_value("173")
        edit = page.get_by_role("button", name="Edit JSON")
        edit.click()
        editor = page.get_by_label("Run configuration (JSON)")
        request = json.loads(editor.input_value())
        assert request["parameters"] == {
            "trees": 24,
            "max_depth": 8,
            "feature_set": "multiscale",
            "sample_weighting": "interval",
        }
        assert request["policy"] == adaptive["default_training_policy"]
        request["policy"] = {"optimizer": "extra_trees", "seeds": [123]}
        request["parameters"] = {"max_depth": 7}
        editor.fill(json.dumps(request))
        page.get_by_role("button", name="Use form").click()
        expect(trees).to_have_value("40")
        expect(depth).to_have_value("7")
        expect(features).to_have_value("greedy")
        expect(weights).to_have_value("per_step")
        expect(page.get_by_label("Random seed", exact=True)).to_have_value("123")
        page.get_by_role("button", name="Start training").click()
        assert api.posts[0][:2] == ("training-runs", request)
    finally:
        page.close()


def test_adaptive_single_validation_is_visible_while_running_and_after_completion(
    browser: Any,
) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.run["training"].update(
        architecture_id="adaptive",
        policy={"budget": "fit", "optimizer": "extra_trees", "seeds": [173]},
    )
    api.run["job"].update(state="running", stage="training", attempt=1)
    api.progress_kind = "fits"
    api.progress_rows = [
        {"seed": 173, "trees": 40, "max_depth": 12, "validation_segment_macro_f1": 0.75}
    ]
    api.metrics.update(
        selected_seed=173, validation_segment_macro_f1=0.75, history={"fits": api.progress_rows}
    )
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(f"{ORIGIN}/training/{RUN_ID}")
        score = page.locator("#training-score")
        expect(score).to_have_text("0.75 validation interval macro-F1 · 40 trees")
        expect(page.locator("#training-charts")).to_be_empty()
        api.run["job"]["state"] = "succeeded"
        page.get_by_role("button", name="Refresh", exact=True).click()
        expect(page.locator("#training-state")).to_have_text("Complete")
        expect(score).to_have_text("0.75 validation interval macro-F1 · 40 trees")
        expect(page.locator("#training-charts")).to_be_empty()
    finally:
        page.close()


@pytest.mark.parametrize("supplemental", [False, True], ids=["training", "saved-weights"])
def test_forest_curve_uses_real_prefix_counts_and_keeps_the_full_model_score(
    browser: Any,
    supplemental: bool,
) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.run["training"].update(
        architecture_id="adaptive",
        policy={"budget": "fit", "optimizer": "extra_trees", "seeds": [173]},
    )
    points = [
        {"trees": trees, "validation_segment_macro_f1": score}
        for trees, score in [(1, 0.89), (5, 0.97), (40, 0.95)]
    ]
    fits = [{**row, "seed": 173, "max_depth": 12} for row in points]
    api.metrics.update(
        selected_seed=173,
        validation_segment_macro_f1=0.95,
        history={"fits": fits[-1:] if supplemental else fits},
    )
    page = browser.new_page(viewport={"width": 320, "height": 800}, locale="en-US")
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        if not supplemental:
            api.run["job"].update(state="running", stage="training", attempt=1)
            api.progress_kind = "fits"
            api.progress_rows = fits[:2]
        page.goto(f"{ORIGIN}/training/{RUN_ID}")
        chart = page.get_by_role("img", name="Validation interval macro-F1 by trees used")
        score = page.locator("#training-score")
        if supplemental:
            expect(score).to_contain_text("0.95")
            expect(chart).to_have_count(0)
            api.metrics["forest_validation_curve"] = {"points": points}
            page.evaluate("window.dispatchEvent(new Event('focus'))")
            expect(page.get_by_text("Measured from saved weights", exact=False)).to_be_visible()
        else:
            expect(score).to_contain_text("0.97 validation interval macro-F1 · 5 trees")
            expect(chart.locator("[data-last] title")).to_have_text("5 trees: 0.97")
            api.metrics["forest_validation_curve"] = {"points": points}
            api.run["job"]["state"] = "succeeded"
            expect(page.locator("#training-state")).to_have_text("Complete", timeout=7000)
            expect(page.get_by_text("Measured from saved weights", exact=False)).to_have_count(0)
        expect(score).to_have_text("0.95 validation interval macro-F1 · 40 trees")
        selection = score.get_by_text("· 40 trees", exact=True)
        for width in (320, 360):
            page.set_viewport_size({"width": width, "height": 800})
            assert selection.evaluate("item => item.getClientRects().length") == 1
        expect(chart).to_have_count(1)
        expect(chart.locator(".training-point")).to_have_count(1)
        expect(chart.locator("[data-last] title")).to_have_text("40 trees: 0.95")
        expect(chart.locator("[data-best]")).to_have_count(0)
        expect(page.get_by_label("Training repeat")).to_be_hidden()
        inspect = page.get_by_role(
            "slider", name="Inspect validation interval macro-f1 by trees used"
        )
        inspect.press("Home")
        expect(inspect).to_have_attribute("aria-valuenow", "1")
        expect(page.locator(".training-chart-tooltip")).to_have_text("0.891 tree")
        inspect.press("ArrowRight")
        expect(inspect).to_have_attribute("aria-valuenow", "5")
        inspect.press("End")
        expect(inspect).to_have_attribute("aria-valuenow", "40")
        inspect.press("Escape")
        page.locator('summary[aria-label="Validation interval macro-F1 values"]').press("Enter")
        table = page.get_by_role("table", name="Validation interval macro-F1 by trees used")
        expect(table.get_by_role("rowheader")).to_have_text(["1", "5", "40"])
        expect(table.get_by_role("cell")).to_have_text(["0.89", "0.97", "0.95"])
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    finally:
        page.close()
