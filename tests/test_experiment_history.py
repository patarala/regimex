"""Grouped history and editable names over real authenticated PostgreSQL records."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any
from uuid import UUID

import pytest
from test_api import (
    PersistentAPI,
    analysis_result_fixture,
    authorization,
    export_fixture,
    persistent_api,
    register_metadata,
    report_metadata_fixture,
    upload,
)

from regimex.api import AnalysisView, create_app
from regimex.application.auth import AuthService
from regimex.auth_contracts import AuthPolicy
from regimex.exports import ExportPayloadV2, seal_export
from regimex.history_contracts import ExperimentName, ExperimentNameUpdate
from regimex.infrastructure.auth_repository import AuthRepository
from regimex.operations import OperationFailure
from regimex.service_contracts import Conflict

# Importing the fixture registers the same isolated schema lifecycle used by API tests.
__all__ = ["persistent_api"]
ORIGIN = "https://regimex.example"


def _publish(api: PersistentAPI, *versions: str) -> None:
    previous = None
    for version in versions:
        model = register_metadata(api, version)
        api.repository.activate_model(
            api.repository.authenticate(api.tokens["admin"]),
            model.model_version_id,
            expected_previous=previous,
        )
        previous = version


def _submit(
    api: PersistentAPI,
    snapshot_id: UUID,
    key: str,
    *,
    model: str = "active",
    owner: str = "alice",
    cancel: bool = False,
) -> AnalysisView:
    response = api.client.post(
        "/api/v1/analyses",
        json={"snapshot_id": str(snapshot_id), "model_version_id": model},
        headers={**api.headers(owner), "Idempotency-Key": key},
    )
    assert response.status_code == 202, response.text
    analysis = AnalysisView.model_validate_json(response.content)
    if cancel:
        cancelled = api.client.post(
            analysis.job.polling_url + "/cancel", headers=api.headers(owner)
        )
        assert cancelled.status_code == 200, cancelled.text
    return analysis


def _complete(api: PersistentAPI, analysis: AnalysisView, tmp_path: Path) -> None:
    operation = api.repository.claim_operation(analysis.analysis_id)
    assert operation is not None and operation.attempt_token is not None
    result = analysis_result_fixture(export_fixture(), analysis.snapshot_id, operation)
    payload = result.model_dump(mode="json")
    report = report_metadata_fixture(
        api,
        tmp_path,
        analysis_id=analysis.analysis_id,
        snapshot_id=analysis.snapshot_id,
        operation=operation,
        result=payload,
    )
    api.repository.complete_analysis(
        analysis.analysis_id, operation.attempt_token, payload, report=report
    )


def _groups(api: PersistentAPI, **parameters: str | int) -> list[dict[str, Any]]:
    response = api.client.get("/api/v1/experiments", params=parameters, headers=api.headers())
    assert response.status_code == 200, response.text
    assert response.headers["Cache-Control"] == "private, no-store"
    items: list[dict[str, Any]] = response.json()["items"]
    return items


def test_group_pages_do_not_split_experiments_or_merge_distinct_snapshots(
    persistent_api: PersistentAPI,
) -> None:
    api = persistent_api
    _publish(api, "group-baseline")
    first = upload(api)
    _submit(api, first, "first", cancel=True)
    # One simulator run can have multiple source snapshots. Only the saved snapshot
    # identity guarantees the identical observations required for comparison.
    payload = export_fixture().model_dump(mode="json", exclude={"content_sha256"})
    payload["source"]["snapshot_id"] += "-second"
    export = seal_export(ExportPayloadV2.model_validate_json(json.dumps(payload)))
    response = api.client.post(
        "/api/v1/snapshots", json=export.model_dump(mode="json"), headers=api.headers()
    )
    assert response.status_code == 201, response.text
    second = UUID(response.json()["snapshot_id"])
    for index in range(23):
        latest_second = _submit(api, second, f"second-{index}", cancel=True)
    third = upload(api, suffix="-third")
    latest_third = _submit(api, third, "third", cancel=True)
    foreign = upload(api, owner="bob")
    _submit(api, foreign, "foreign", owner="bob", cancel=True)

    page = _groups(api, limit=2, offset=0)
    following = _groups(api, limit=2, offset=2)
    assert [item["snapshot_id"] for item in page + following] == [
        str(third),
        str(second),
        str(first),
    ]
    assert [item["analysis_count"] for item in page + following] == [1, 23, 1]
    assert [item["matching_analysis_count"] for item in page + following] == [1, 23, 1]
    assert all(item["successful_analysis_count"] == 0 for item in page + following)
    assert page[0]["latest_analysis"]["analysis_id"] == str(latest_third.analysis_id)
    assert page[1]["latest_analysis"]["analysis_id"] == str(latest_second.analysis_id)
    assert page[1]["run_id"] == following[0]["run_id"]
    assert _groups(api, limit=2, offset=3) == []
    children = api.client.get(
        "/api/v1/analyses", params={"snapshot_id": str(second)}, headers=api.headers()
    ).json()["items"]
    assert len(children) == 23
    assert all(item["snapshot_id"] == str(second) for item in children)


def test_group_filters_count_all_runs_but_select_latest_matching_run(
    persistent_api: PersistentAPI,
    tmp_path: Path,
) -> None:
    api = persistent_api
    _publish(api, "first-model", "second-model")
    snapshot_id = upload(api)
    succeeded = _submit(api, snapshot_id, "success", model="first-model")
    _complete(api, succeeded, tmp_path)
    _submit(api, snapshot_id, "cancelled", model="first-model", cancel=True)
    failed = _submit(api, snapshot_id, "failed", model="second-model")
    claimed = api.repository.claim_operation(failed.analysis_id)
    assert claimed is not None and claimed.attempt_token is not None
    api.repository.fail_operation(
        failed.analysis_id,
        claimed.attempt_token,
        OperationFailure(code="invalid_data", message="Deliberate fixture failure."),
    )
    _submit(api, snapshot_id, "queued", model="second-model")
    cancel_requested = _submit(api, snapshot_id, "cancelling", model="second-model")
    assert api.repository.claim_operation(cancel_requested.analysis_id) is not None
    api.repository.cancel_operation(
        api.repository.authenticate(api.tokens["alice"]), cancel_requested.analysis_id
    )
    running = _submit(api, snapshot_id, "running", model="first-model")
    assert api.repository.claim_operation(running.analysis_id) is not None
    unrelated = upload(api, suffix="-unrelated")
    _submit(api, unrelated, "newest-unrelated", model="first-model", cancel=True)

    for parameters, matching, latest in (
        ({"q": "SECOND-MODEL", "state": "active"}, 2, cancel_requested),
        ({"state": "succeeded"}, 1, succeeded),
        ({"q": "first-model", "state": "running"}, 1, running),
        ({"state": "failed"}, 1, failed),
    ):
        groups = _groups(api, **parameters, limit=1)
        assert len(groups) == 1
        group = groups[0]
        assert group["snapshot_id"] == str(snapshot_id)
        assert group["analysis_count"] == 6
        assert group["successful_analysis_count"] == 1
        assert group["matching_analysis_count"] == matching
        assert group["latest_analysis"]["analysis_id"] == str(latest.analysis_id)
        children = api.client.get(
            "/api/v1/analyses",
            params={**parameters, "snapshot_id": str(snapshot_id)},
            headers=api.headers(),
        ).json()["items"]
        assert len(children) == matching
        assert children[0] == group["latest_analysis"]
    assert _groups(api, q="not present", state="succeeded") == []


def test_names_are_searchable_literal_metadata_and_reupload_preserves_them(
    persistent_api: PersistentAPI,
) -> None:
    api = persistent_api
    _publish(api, "name-baseline")
    snapshot_id = upload(api)
    _submit(api, snapshot_id, "named", cancel=True)
    _submit(api, snapshot_id, "named-again", cancel=True)
    other = upload(api, suffix="-other")
    _submit(api, other, "other", cancel=True)
    path = f"/api/v1/experiments/{snapshot_id}/name"
    name = "Рынок 100%_test <b> & recovery Straße café"
    response = api.client.patch(
        path,
        json={"display_name": "  " + name.replace("é", "e\u0301") + "  ", "expected_revision": 0},
        headers=api.headers(),
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"snapshot_id": str(snapshot_id), "display_name": name, "revision": 1}
    assert upload(api) == snapshot_id
    for query in ("РЫНОК", "100%_test", "<b>", "& recovery", "STRASSE", "cafe\u0301"):
        groups = _groups(api, q=query)
        assert len(groups) == 1, query
        assert groups[0]["display_name"] == name
        assert groups[0]["name_revision"] == 1
        assert groups[0]["matching_analysis_count"] == 2
        children = api.client.get(
            "/api/v1/analyses", params={"q": query}, headers=api.headers()
        ).json()["items"]
        assert len(children) == 2
        assert all(item["snapshot_id"] == str(snapshot_id) for item in children)
    assert _groups(api, q="' OR 1=1 --") == []
    stale = api.client.patch(
        path, json={"display_name": "stale change", "expected_revision": 0}, headers=api.headers()
    )
    assert stale.status_code == 409, stale.text
    reset = api.client.patch(
        path, json={"display_name": None, "expected_revision": 1}, headers=api.headers()
    )
    assert reset.status_code == 200, reset.text
    assert reset.json()["display_name"] is None and reset.json()["revision"] == 2
    assert _groups(api, q="100%_test") == []


def test_concurrent_renames_cannot_overwrite_an_unseen_name(
    persistent_api: PersistentAPI,
) -> None:
    api = persistent_api
    snapshot_id = upload(api)
    path = f"/api/v1/experiments/{snapshot_id}/name"
    barrier = Barrier(2)

    principal = api.repository.authenticate(api.tokens["alice"])

    def rename(name: str) -> ExperimentName | Conflict:
        barrier.wait(timeout=10)
        try:
            return api.service.rename_experiment(
                principal,
                snapshot_id,
                ExperimentNameUpdate(display_name=name, expected_revision=0),
            )
        except Conflict as failure:
            return failure

    with ThreadPoolExecutor(max_workers=2) as executor:
        replies = list(executor.map(rename, ("First title", "Second title")))
    assert sum(isinstance(reply, Conflict) for reply in replies) == 1
    winner = next(reply for reply in replies if isinstance(reply, ExperimentName))
    reset = api.client.patch(
        path,
        json={"display_name": None, "expected_revision": winner.revision},
        headers=api.headers(),
    )
    assert reset.status_code == 200, reset.text
    assert reset.json()["revision"] == 2


def test_renaming_preserves_committed_snapshot_results_and_report_bytes(
    persistent_api: PersistentAPI,
    tmp_path: Path,
) -> None:
    api = persistent_api
    _publish(api, "immutable-baseline")
    snapshot_id = upload(api)
    analysis = _submit(api, snapshot_id, "immutable")
    _complete(api, analysis, tmp_path)
    urls = (f"/api/v1/snapshots/{snapshot_id}/export", analysis.result_url, analysis.report_url)
    before = [api.client.get(url, headers=api.headers()).content for url in urls]
    with api.repository.connection() as connection:
        row_before = connection.execute(
            "SELECT s.document::text,s.content_sha256,s.object_key,a.result::text,o.document::text "
            "FROM regimex_snapshots s JOIN regimex_analyses a USING(snapshot_id) "
            "JOIN regimex_operations o USING(operation_id) WHERE a.analysis_id=%s",
            (analysis.analysis_id,),
        ).fetchone()
    renamed = api.client.patch(
        f"/api/v1/experiments/{snapshot_id}/name",
        json={"display_name": "Readable experiment", "expected_revision": 0},
        headers=api.headers(),
    )
    assert renamed.status_code == 200, renamed.text
    with api.repository.connection() as connection:
        row_after = connection.execute(
            "SELECT s.document::text,s.content_sha256,s.object_key,a.result::text,o.document::text "
            "FROM regimex_snapshots s JOIN regimex_analyses a USING(snapshot_id) "
            "JOIN regimex_operations o USING(operation_id) WHERE a.analysis_id=%s",
            (analysis.analysis_id,),
        ).fetchone()
    assert row_after == row_before
    assert [api.client.get(url, headers=api.headers()).content for url in urls] == before


@pytest.mark.parametrize(
    "body",
    [
        {"display_name": "", "expected_revision": 0},
        {"display_name": "   ", "expected_revision": 0},
        {"display_name": "bad\nname", "expected_revision": 0},
        {"display_name": "bad\x00name", "expected_revision": 0},
        {"display_name": "x" * 121, "expected_revision": 0},
        {"display_name": "\u0344" * 120, "expected_revision": 0},
        {"display_name": "valid", "expected_revision": -1},
        {"display_name": "valid", "expected_revision": "0"},
        {"display_name": "valid"},
    ],
)
def test_rename_rejects_invalid_metadata(
    persistent_api: PersistentAPI,
    body: dict[str, object],
) -> None:
    api = persistent_api
    snapshot_id = upload(api)
    response = api.client.patch(
        f"/api/v1/experiments/{snapshot_id}/name", json=body, headers=api.headers()
    )
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "invalid_data"


def test_names_require_owner_access_before_reporting_revision_conflicts(
    persistent_api: PersistentAPI,
) -> None:
    api = persistent_api
    _publish(api, "owner-baseline")
    snapshot_id = upload(api)
    _submit(api, snapshot_id, "owner", cancel=True)
    path = f"/api/v1/experiments/{snapshot_id}/name"
    body = {"display_name": "Private experiment", "expected_revision": 0}
    assert api.client.patch(path, json=body).status_code == 401
    assert api.client.get("/api/v1/experiments").status_code == 401
    assert api.client.patch(path, json=body, headers=api.headers()).status_code == 200
    for owner in ("bob", "admin", "integration"):
        denied = api.client.patch(path, json=body, headers=api.headers(owner))
        assert denied.status_code == 404, denied.text
        groups = api.client.get(
            "/api/v1/experiments", params={"q": "Private"}, headers=api.headers(owner)
        )
        assert groups.status_code == 200 and groups.json()["items"] == []


def test_browser_rename_requires_same_origin_and_session_csrf(
    persistent_api: PersistentAPI,
) -> None:
    api = persistent_api
    snapshot_id = upload(api)
    from fastapi.testclient import TestClient

    authentication = AuthService(AuthRepository(api.repository), AuthPolicy(public_origin=ORIGIN))
    with TestClient(
        create_app(api.service, auth_service=authentication), base_url=ORIGIN
    ) as client:
        login = client.post(
            "/api/v1/session", json={"token": api.tokens["alice"]}, headers={"Origin": ORIGIN}
        )
        assert login.status_code == 200, login.text
        csrf = login.json()["csrf_token"]
        path = f"/api/v1/experiments/{snapshot_id}/name"
        body = {"display_name": "Session title", "expected_revision": 0}
        for headers in (
            {},
            {"Origin": ORIGIN},
            {"Origin": "https://foreign.example", "X-CSRF-Token": csrf},
            {"Origin": ORIGIN, "X-CSRF-Token": "0" * 64},
        ):
            assert client.patch(path, json=body, headers=headers).status_code == 403
        accepted = client.patch(path, json=body, headers={"Origin": ORIGIN, "X-CSRF-Token": csrf})
        assert accepted.status_code == 200, accepted.text
        assert accepted.json()["revision"] == 1


def test_delegated_history_and_names_respect_subject_source_and_run(
    persistent_api: PersistentAPI,
) -> None:
    api = persistent_api
    _publish(api, "delegated-baseline")
    from fastapi.testclient import TestClient

    authentication = AuthService(AuthRepository(api.repository), AuthPolicy(public_origin=ORIGIN))
    with TestClient(
        create_app(api.service, auth_service=authentication), base_url=ORIGIN
    ) as client:
        run_id = export_fixture().source.run_id
        headers = {
            **api.headers("integration"),
            "X-Regimex-Subject": "external-alice",
            "X-Regimex-Run-Id": run_id,
        }
        uploaded = client.post(
            "/api/v1/integration/snapshots",
            json=export_fixture().model_dump(mode="json"),
            headers=headers,
        )
        assert uploaded.status_code == 201, uploaded.text
        snapshot_id = UUID(uploaded.json()["snapshot_id"])
        accepted = client.post(
            "/api/v1/integration/analyses",
            json={"snapshot_id": str(snapshot_id)},
            headers={**headers, "Idempotency-Key": "delegated"},
        )
        assert accepted.status_code == 202, accepted.text
        path = f"/api/v1/integration/experiments/{snapshot_id}/name"
        body = {"display_name": "Scoped title", "expected_revision": 0}
        renamed = client.patch(path, json=body, headers=headers)
        assert renamed.status_code == 200, renamed.text
        visible = client.get("/api/v1/integration/experiments", headers=headers)
        assert visible.status_code == 200, visible.text
        assert visible.json()["items"][0]["analysis_count"] == 1
        assert visible.json()["items"][0]["latest_analysis"]["polling_url"].startswith(
            "/api/v1/integration/analyses/"
        )
        for changed in (
            {"X-Regimex-Subject": "external-bob"},
            {"X-Regimex-Run-Id": "another-run"},
        ):
            outside = {**headers, **changed}
            assert client.patch(path, json=body, headers=outside).status_code == 404
            response = client.get("/api/v1/integration/experiments", headers=outside)
            assert response.status_code == 200 and response.json()["items"] == []
        integration = api.repository.authenticate(api.tokens["integration"])
        other_source_token = api.repository.create_token(
            integration.owner_id, "integration", source_namespace="another-source"
        )
        other_source_headers = {**headers, **authorization(other_source_token)}
        response = client.get("/api/v1/integration/experiments", headers=other_source_headers)
        assert response.status_code == 200 and response.json()["items"] == []
        assert client.patch(path, json=body, headers=other_source_headers).status_code == 404
        unchanged = client.get("/api/v1/integration/experiments", headers=headers)
        assert unchanged.status_code == 200
        assert unchanged.json()["items"][0]["display_name"] == "Scoped title"


@pytest.mark.parametrize(
    ("name", "query"),
    [
        ("Αγορά μετοχών", "ΑΓΟΡΆ ΜΕΤΟΧΏΝ"),
        ("市场实验", "市场"),
        ("تجربة السوق", "السوق"),
    ],
)
def test_name_search_supports_unicode_without_database_locale_rules(
    persistent_api: PersistentAPI,
    name: str,
    query: str,
) -> None:
    api = persistent_api
    _publish(api, "unicode-baseline")
    snapshot_id = upload(api)
    analysis = _submit(api, snapshot_id, "unicode-name", cancel=True)
    response = api.client.patch(
        f"/api/v1/experiments/{snapshot_id}/name",
        json={"display_name": name, "expected_revision": 0},
        headers=api.headers(),
    )
    assert response.status_code == 200, response.text
    groups = _groups(api, q=query)
    assert len(groups) == 1 and groups[0]["display_name"] == name
    assert groups[0]["latest_analysis"]["analysis_id"] == str(analysis.analysis_id)
    children = api.client.get(
        "/api/v1/analyses", params={"q": query}, headers=api.headers()
    ).json()["items"]
    assert [item["analysis_id"] for item in children] == [str(analysis.analysis_id)]
