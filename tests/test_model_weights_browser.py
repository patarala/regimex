"""Weight cleanup preserves the run and can resume after leaving Training."""

from typing import Any
from uuid import UUID

import pytest
from test_offline_report_browser import browser as browser
from test_training_workspace_browser import (
    EVALUATION_ID,
    ORIGIN,
    PUBLICATION_ID,
    RUN_ID,
    TrainingAPI,
    completed_run,
)


def test_discard_weights_and_resume_cleanup_with_metrics_and_evaluation(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.runs.append(api.run)
    api.actions = [completed_run(EVALUATION_ID, "evaluation")]
    deletion_id = str(UUID(int=10))
    receipt = {
        "deletion_id": deletion_id,
        "sha256": "a" * 64,
        "target_id": RUN_ID,
        "target_kind": "model_weights",
        "state": "pending",
        "pending_bytes": 8192,
        "reclaimed_bytes": 0,
    }
    plan = {
        "plan_id": deletion_id,
        "sha256": receipt["sha256"],
        "target_kind": "model_weights",
        "expires_at": "2099-01-01T00:00:00Z",
        "size_bytes": 8192,
        "blockers": [{"message": "An evaluation is still running."}],
    }
    submissions: list[str] = []

    def preview(request: Any) -> None:
        assert request.request.method == "POST"
        assert request.request.headers["x-csrf-token"] == "test"
        request.fulfill(json=plan)

    def deletion(request: Any) -> None:
        if request.request.method == "POST":
            assert request.request.headers["x-csrf-token"] == "test"
            expected_body = {"sha256": receipt["sha256"]}
            if not submissions:
                expected_body["archive_run"] = True
            assert request.request.post_data_json == expected_body
            submissions.append(request.request.url)
            if len(submissions) == 1:
                api.run["weights_deletion_id"] = deletion_id
                api.run["archived_at"] = "2026-11-21T12:00:00Z"
            else:
                receipt.update(state="completed", reclaimed_bytes=8192, pending_bytes=0)
        request.fulfill(json=receipt)

    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    page.route(f"**/training-runs/{RUN_ID}/weights/deletion-preview", preview)
    page.route(f"**/deletions/{deletion_id}", deletion)
    page.route(
        "**/api/v1/deletions?*",
        lambda route: route.fulfill(
            json={"items": [receipt] if submissions and receipt["state"] == "pending" else []}
        ),
    )
    try:
        page.goto(f"{ORIGIN}/training/{RUN_ID}")
        expect(page.get_by_role("button", name="Use as default")).to_be_visible()
        page.get_by_text("Run details and settings", exact=True).click()
        page.get_by_role("button", name="Delete model weights").press("Enter")
        dialog = page.get_by_role("dialog")
        confirm = dialog.get_by_role("button", name="Delete", exact=True)
        archive = dialog.get_by_role("checkbox", name="Also archive this run")
        expect(confirm).to_be_disabled()
        expect(archive).to_be_hidden()
        expect(dialog).to_contain_text("An evaluation is still running.")
        plan["blockers"] = []
        dialog.get_by_role("button", name="Refresh preview").press("Enter")
        expect(dialog).to_contain_text("8 KiB")
        expect(dialog).to_contain_text("You can still view training metrics and saved evaluations")
        expect(archive).not_to_be_checked()
        archive.check()
        confirm.press("Enter")
        expect(dialog).to_contain_text("waiting for cleanup")
        expect(archive).to_be_hidden()
        dialog.get_by_role("button", name="Close").press("Enter")
        expect(page.get_by_role("button", name="Use as default")).to_be_hidden()
        expect(page.get_by_role("button", name="Delete model weights")).to_be_hidden()
        expect(page.locator("#training-score")).to_contain_text("0.72")
        expect(page.locator(".evaluation-summary")).to_be_visible()

        page.get_by_role("link", name="Analyses", exact=True).click()
        page.get_by_text("Pending deletions", exact=True).click()
        page.get_by_role("button", name="Model weights 00000001 · 8 KiB pending").click()
        dialog.get_by_role("button", name="Retry cleanup").press("Enter")
        expect(dialog).to_be_hidden()
        expect(page.get_by_text("Deleted model weights. Released 8 KiB.")).to_be_visible()
        page.go_back()
        expect(page.get_by_role("button", name="Evaluate model")).to_be_hidden()
        expect(page.get_by_role("button", name="Use as default")).to_be_hidden()
        expect(page.locator("#training-score")).to_contain_text("0.72")
        expect(page.locator(".evaluation-summary")).to_be_visible()
        page.get_by_role("button", name="View deletion receipt").press("Enter")
        expect(dialog).to_contain_text("Deletion complete")
        expect(dialog).to_contain_text("8 KiB released.")
        assert len(submissions) == 2
    finally:
        page.close()


def test_previously_published_model_has_no_delete_action(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.runs.append(api.run)
    api.actions = [completed_run(PUBLICATION_ID, "publication")]
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(f"{ORIGIN}/training/{RUN_ID}")
        expect(page.locator("#training-actions")).to_contain_text("Publication")
        page.get_by_text("Run details and settings", exact=True).click()
        expect(page.get_by_role("button", name="Delete model weights")).to_be_hidden()
    finally:
        page.close()


def test_leaving_run_during_confirmation_keeps_the_receipt_reachable(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.runs.append(api.run)
    deletion_id = str(UUID(int=10))
    receipt = {
        "deletion_id": deletion_id,
        "sha256": "f" * 64,
        "target_id": RUN_ID,
        "target_kind": "model_weights",
        "state": "pending",
        "pending_bytes": 8192,
    }
    confirmations: list[Any] = []
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    page.route(
        f"**/training-runs/{RUN_ID}/weights/deletion-preview",
        lambda route: route.fulfill(
            json={
                **receipt,
                "plan_id": deletion_id,
                "size_bytes": 8192,
                "expires_at": "2099-01-01T00:00:00Z",
                "blockers": [],
            }
        ),
    )
    page.route(f"**/deletions/{deletion_id}", lambda route: confirmations.append(route))
    page.route(
        "**/api/v1/deletions?*",
        lambda route: route.fulfill(json={"items": [receipt] if confirmations else []}),
    )
    try:
        page.goto(ORIGIN)
        page.get_by_role("link", name="Training", exact=True).click()
        page.get_by_text("Run details and settings", exact=True).click()
        page.get_by_role("button", name="Delete model weights").click()
        dialog = page.get_by_role("dialog")
        with page.expect_request(lambda request: request.url.endswith(deletion_id)):
            dialog.get_by_role("button", name="Delete", exact=True).click()
        page.go_back()
        expect(dialog).to_be_hidden()
        assert confirmations[0].request.post_data_json == {"sha256": receipt["sha256"]}
        api.run["weights_deletion_id"] = deletion_id
        confirmations[0].fulfill(json=receipt)
        page.get_by_text("Pending deletions", exact=True).click()
        expect(
            page.get_by_role("button", name="Model weights 00000001 · 8 KiB pending")
        ).to_be_visible()
        page.go_forward()
        expect(page.get_by_role("button", name="View deletion receipt")).to_be_visible()
        expect(page.get_by_role("button", name="Evaluate model")).to_be_hidden()
    finally:
        page.close()
