"""Imported training evidence shares the service run history and navigation."""

from copy import deepcopy
from typing import Any
from uuid import UUID

import pytest
from test_offline_report_browser import browser as browser
from test_training_workspace_browser import ORIGIN, RUN_ID, TrainingAPI

IMPORTED = str(UUID(int=40))


def imported_run() -> dict[str, Any]:
    reference = {
        "reference_set_id": "offline-validation",
        "content_sha256": "a" * 64,
        "direction_semantics_sha256": "b" * 64,
    }
    evaluation = {
        "candidate": "forest",
        "interval_macro_f1": 0.75,
        "point_macro_f1": 0.83,
        "sample_count": 48,
        "passed": True,
        "bundle_sha256": "9" * 64,
        "report_sha256": "6" * 64,
        "freeze_sha256": "5" * 64,
        "protocol_sha256": "4" * 64,
    }
    return {
        "run_id": IMPORTED,
        "name_revision": 0,
        "display_name": None,
        "job": {"state": "succeeded", "stage": "imported", "created_at": "2026-01-02T10:00:00Z"},
        "training": {
            "origin": "external",
            "model_version_id": "forest-v1",
            "architecture_id": "adaptive",
            "build_id": "source-e58b",
            "analysis_profile_id": "direction-32/2.2",
            "evidence": {
                "run_id": "offline-fit-3",
                "dataset_sha256": "1" * 64,
                "split_manifest_sha256": "2" * 64,
                "validation_references": reference,
                "labels": {**reference, "reference_set_id": "offline-training"},
            },
            "weights": {"path": "weights.npz", "size_bytes": 1024, "sha256": "3" * 64},
            "configuration": {"parameters": {"seed": 9007199254740993}},
            "external_evaluations": [
                {**evaluation, "kind": "native_final"},
                {
                    **evaluation,
                    "kind": "controlled_transfer",
                    "point_macro_f1": None,
                    "sample_count": 12,
                    "passed": False,
                    "interval_macro_f1": 0,
                },
            ],
            "external_latency": {
                "manifest_sha256": "9" * 64,
                "report_sha256": "7" * 64,
                "threads": 2,
                "timings": [{"steps": 256, "predict_ms": None, "predict_decode_ms": 4.25}],
            },
        },
        "output": {"bundle_sha256": "9" * 64, "validation_segment_macro_f1": 0.66},
    }


def test_imported_model_keeps_its_evidence_and_existing_run_controls(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.runs = [imported_run(), api.run]
    page = browser.new_page(viewport={"width": 320, "height": 760}, locale="en-US")
    page.route(ORIGIN + "/**", api.dispatch)
    page.route(
        f"**/training-runs/{IMPORTED}/metrics",
        lambda route: route.fulfill(
            json={
                "validation_segment_macro_f1": 0.66,
                "selected_seed": 11,
                "history": {"fits": [{"seed": 11, "validation_segment_macro_f1": 0.66}]},
            }
        ),
    )
    try:
        page.goto(ORIGIN + "/training")
        expect(page).to_have_url(f"{ORIGIN}/training/{IMPORTED}")
        expect(page.locator("#training-detail-title")).to_contain_text("forest-v1")
        expect(page.locator("#training-state")).to_have_text("Imported")
        expect(page.locator("#training-created time")).to_contain_text("Imported")
        expect(page.locator("#training-score")).to_contain_text("0.66 validation interval macro-F1")
        for selector in (
            "#training-copy-run",
            "#training-timing",
            "#training-cancel",
            "#training-downstream",
            "#training-charts figure",
        ):
            expect(page.locator(selector)).to_be_hidden()
        results = page.locator("#training-external-evaluations")
        expect(results.get_by_role("heading", name="Native final benchmark")).to_be_visible()
        expect(results.locator("section").nth(0)).to_contain_text("Interval macro-F10.75")
        transfer = results.locator("section").nth(1)
        expect(transfer).to_contain_text("Interval macro-F10")
        expect(transfer).to_contain_text("Paired paths12")
        expect(transfer).to_contain_text("Quality gateFailed")
        expect(transfer).not_to_contain_text("Per-step")
        latency = results.get_by_role("table", name="Median inference latency (ms)")
        expect(latency.get_by_role("columnheader")).to_have_text(["Steps", "Prediction + decoding"])
        expect(latency.get_by_role("cell")).to_have_text(["256", "4.25"])
        transfer.get_by_text("Evaluation provenance").click()
        expect(transfer).to_contain_text("6" * 64)
        page.get_by_text("Bundle, weights and settings", exact=True).click()
        expect(page.locator("#training-facts")).to_contain_text("weights.npz · 1 KiB · saved")
        expect(page.locator("#training-settings")).to_contain_text("9007199254740993")
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert not any(path.startswith(f"training-runs/{IMPORTED}/actions") for path in api.pages)
        page.get_by_role("button", name="Archive run", exact=True).click()
        expect(page.get_by_role("button", name="Restore run")).to_be_enabled()
        page.get_by_role("button", name="Run history", exact=True).click()
        view = page.get_by_role("combobox", name="Run history view")
        view.select_option("all")
        row = page.locator(f'button[data-run-id="{IMPORTED}"]')
        expect(row).to_contain_text("forest-v1")
        expect(row.locator("[data-state]")).to_have_text("Imported")
        page.locator(f'button[data-run-id="{RUN_ID}"]').click()
        expect(page.get_by_role("button", name="New run from this run")).to_be_visible()
        expect(results).to_be_hidden()
        api.run["external_evaluations"] = api.runs[0]["training"]["external_evaluations"]
        api.run["external_latency"] = api.runs[0]["training"]["external_latency"]
        page.evaluate("window.dispatchEvent(new Event('focus'))")
        expect(latency).to_be_visible()
        expect(page.get_by_role("button", name="Evaluate model")).to_be_visible()
        expect(page.locator("#training-state")).to_have_text("Complete")
        page.go_back()
        page.get_by_role("button", name="Run history", exact=True).click()
        expect(view).to_have_value("all")
        expect(page.locator("#training-state")).to_have_text("Imported")
        expect(results).to_be_visible()
        later = deepcopy(api.runs[0])
        later.update(run_id=str(UUID(int=41)), display_name="Another terminal fit")
        api.runs.insert(0, later)
        page.evaluate("window.dispatchEvent(new Event('focus'))")
        expect(page.locator("#training-runs")).to_contain_text("Another terminal fit")
        expect(page.locator("#training-detail-title")).to_contain_text("forest-v1")
    finally:
        page.close()


def test_import_comparison_checks_validation_evidence_and_preserves_exact_settings(
    browser: Any,
) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    first = imported_run()
    second = deepcopy(first)
    second["run_id"] = RUN_ID
    second["training"]["model_version_id"] = "forest-v2"
    second["training"]["configuration"]["parameters"]["seed"] += 1
    second["output"]["validation_segment_macro_f1"] = 0.7
    api.run, api.runs = second, [first, second]
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(f"{ORIGIN}/training/compare?left={IMPORTED}&right={RUN_ID}")
        status = page.locator("#training-comparison-status")
        expect(status).to_contain_text("+0.0400")
        expect(page.get_by_role("row", name="Status")).to_contain_text("Imported")
        expect(page.get_by_role("row", name="parameters.seed")).to_contain_text("9007199254740993")
        second["training"]["evidence"]["validation_references"]["content_sha256"] = "8" * 64
        page.reload()
        expect(status).to_contain_text("Comparing validation scores requires the same")
        expect(page.get_by_role("row", name="Validation interval macro-F1")).to_have_count(0)
    finally:
        page.close()
