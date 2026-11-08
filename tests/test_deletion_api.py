"""Deletion authorization, explicit confirmation and retry over real persistence."""

from collections.abc import Iterator
from uuid import UUID

import pytest

pytest.importorskip("psycopg", reason="install the service extra")
pytest.importorskip("httpx2", reason="install the service-test group")
pytest.importorskip("procrastinate", reason="install the queue-test group")

from fastapi.testclient import TestClient
from test_integration_scope import ORIGIN, PREFIX, IntegrationContext, export, upload
from test_integration_scope import context as context
from test_integration_security import key

from regimex.api import create_app
from regimex.application.deletion import DeletionService
from regimex.infrastructure.deletion_repository import DeletionRepository
from regimex.infrastructure.object_store import ObjectInfo


@pytest.fixture
def client(context: IntegrationContext) -> Iterator[TestClient]:
    application = create_app(
        context.service,
        auth_service=context.authentication,
        deletion_service=DeletionService(
            DeletionRepository(context.database), context.service.store
        ),
    )
    with TestClient(application, base_url=ORIGIN) as value:
        yield value


def test_deletion_is_opt_in_and_bound_to_owner_source_and_run(
    context: IntegrationContext,
    client: TestClient,
) -> None:
    snapshot_id = upload(context)
    path = f"{PREFIX}/experiments/{snapshot_id}/deletion-preview"
    for credential in (context.gateway_token, key(context, ("read",)), key(context, ("upload",))):
        headers = context.headers(token=credential)
        assert client.post(path, headers=headers).status_code == 403
        assert client.get(PREFIX + "/deletions", headers=headers).status_code == 403
    deleter = key(context, ("delete",))
    for headers in (
        context.headers("bob", token=deleter),
        context.headers(run="other", token=deleter),
    ):
        assert client.post(path, headers=headers).status_code == 404
    headers = context.headers(token=deleter)
    preview = client.post(path, headers=headers)
    assert preview.status_code == 200, preview.text
    assert preview.headers["cache-control"] == "private, no-store"
    plan = preview.json()
    assert plan["blockers"] == [] and plan["experiment_count"] == 1
    assert plan["analysis_count"] == 0 and plan["object_count"] == 1
    assert (
        plan["size_bytes"] == context.database.get_snapshot_internal(snapshot_id).object.size_bytes
    )
    assert "object_key" not in preview.text and "version_id" not in preview.text
    target = f"{PREFIX}/deletions/{plan['plan_id']}"
    assert client.post(target, json={}, headers=headers).status_code == 422
    assert client.post(target, json={"sha256": "0" * 64}, headers=headers).status_code == 409
    accepted = client.post(target, json={"sha256": plan["sha256"]}, headers=headers)
    assert accepted.status_code == 200, accepted.text
    receipt = accepted.json()
    assert receipt["state"] == "completed" and receipt["pending_bytes"] == 0
    assert receipt["reclaimed_bytes"] == plan["size_bytes"]
    assert client.get(target, headers=headers).json() == receipt
    assert client.get(target, headers=context.headers("bob", token=deleter)).status_code == 404
    assert (
        client.get(target, headers=context.headers(run="other", token=deleter)).status_code == 404
    )
    assert client.post(target, json={"sha256": plan["sha256"]}, headers=headers).json() == receipt
    assert client.get(PREFIX + "/deletions", headers=headers).json()["items"] == []
    assert (
        context.client.get(
            f"{PREFIX}/snapshots/{snapshot_id}", headers=context.headers()
        ).status_code
        == 404
    )
    repeated_upload = context.client.post(
        PREFIX + "/snapshots", json=export().model_dump(mode="json"), headers=context.headers()
    )
    assert repeated_upload.status_code == 409
    assert (
        context.client.get(PREFIX + "/storage", headers=context.headers()).json()["used_bytes"] == 0
    )


def test_failed_physical_cleanup_remains_charged_and_can_resume_after_reopening(
    context: IntegrationContext,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_id = upload(context)
    headers = context.headers(token=key(context, ("delete",)))
    preview = client.post(f"{PREFIX}/experiments/{snapshot_id}/deletion-preview", headers=headers)
    assert preview.status_code == 200, preview.text
    plan = preview.json()
    target = f"{PREFIX}/deletions/{plan['plan_id']}"
    original = context.service.store.delete_version

    def unavailable(_expected: ObjectInfo) -> None:
        raise OSError("Deliberate unavailable fixture storage")

    monkeypatch.setattr(context.service.store, "delete_version", unavailable)
    response = client.post(target, json={"sha256": plan["sha256"]}, headers=headers)
    assert response.status_code == 202, response.text
    pending = response.json()
    assert pending["state"] == "pending" and pending["pending_objects"] == 1
    assert pending["pending_bytes"] == plan["size_bytes"] and pending["reclaimed_bytes"] == 0
    assert pending["failure"] and "Deliberate unavailable" not in pending["failure"]
    listed = client.get(PREFIX + "/deletions", headers=headers).json()["items"]
    assert len(listed) == 1 and listed[0]["deletion_id"] == plan["plan_id"]
    usage = context.client.get(PREFIX + "/storage", headers=context.headers()).json()
    assert usage["used_bytes"] + usage["reserved_bytes"] == plan["size_bytes"]
    monkeypatch.setattr(context.service.store, "delete_version", original)
    # A fresh service instance resumes the persisted receipt, without original UI state.
    service = DeletionService(DeletionRepository(context.database), context.service.store)
    with TestClient(
        create_app(context.service, auth_service=context.authentication, deletion_service=service),
        base_url=ORIGIN,
    ) as reopened:
        retried = reopened.post(target, json={"sha256": listed[0]["sha256"]}, headers=headers)
    assert retried.status_code == 200, retried.text
    assert retried.json()["state"] == "completed"
    usage = context.client.get(PREFIX + "/storage", headers=context.headers()).json()
    assert usage["used_bytes"] == usage["reserved_bytes"] == 0


def test_browser_deletion_requires_session_csrf_for_preview_and_confirmation(
    context: IntegrationContext,
    client: TestClient,
) -> None:
    user_headers = {"Authorization": "Bearer " + context.user_token}
    uploaded = client.post(
        "/api/v1/snapshots", json=export().model_dump(mode="json"), headers=user_headers
    )
    assert uploaded.status_code == 201
    snapshot_id = UUID(uploaded.json()["snapshot_id"])
    login = client.post(
        "/api/v1/session", json={"token": context.user_token}, headers={"Origin": ORIGIN}
    )
    assert login.status_code == 200
    csrf = login.json()["csrf_token"]
    path = f"/api/v1/experiments/{snapshot_id}/deletion-preview"
    for headers in (
        {},
        {"Origin": ORIGIN},
        {"Origin": "https://foreign.example", "X-CSRF-Token": csrf},
    ):
        assert client.post(path, headers=headers).status_code == 403
    headers = {"Origin": ORIGIN, "X-CSRF-Token": csrf}
    preview = client.post(path, headers=headers)
    assert preview.status_code == 200, preview.text
    plan = preview.json()
    target = f"/api/v1/deletions/{plan['plan_id']}"
    assert client.post(target, json={"sha256": plan["sha256"]}).status_code == 403
    response = client.post(target, json={"sha256": plan["sha256"]}, headers=headers)
    assert response.status_code == 200, response.text
