"""Snapshots remain discoverable before analysis and after removing their runs."""

from test_api import PersistentAPI, upload
from test_api import persistent_api as persistent_api
from test_integration_scope import PREFIX, IntegrationContext
from test_integration_scope import context as context
from test_integration_scope import upload as delegated_upload


def test_empty_experiment_is_owned_searchable_and_has_zero_runs(
    persistent_api: PersistentAPI,
) -> None:
    api = persistent_api
    snapshot_id = upload(api)
    upload(api, owner="bob")
    response = api.client.get("/api/v1/experiments", headers=api.headers())
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "private, no-store"
    items = response.json()["items"]
    assert len(items) == 1
    assert items[0]["snapshot_id"] == str(snapshot_id)
    assert items[0]["latest_analysis"] is None
    assert items[0]["analysis_count"] == 0
    assert items[0]["successful_analysis_count"] == 0
    assert items[0]["matching_analysis_count"] == 0

    named = api.client.patch(
        f"/api/v1/experiments/{snapshot_id}/name",
        json={"display_name": "Unanalysed experiment", "expected_revision": 0},
        headers=api.headers(),
    )
    assert named.status_code == 200
    for query, count in (("unanalysed", 1), ("unknown-name", 0)):
        result = api.client.get("/api/v1/experiments", params={"q": query}, headers=api.headers())
        assert result.status_code == 200
        assert len(result.json()["items"]) == count
    for state in ("active", "succeeded", "cancelled", "failed"):
        result = api.client.get(
            "/api/v1/experiments", params={"state": state}, headers=api.headers()
        )
        assert result.status_code == 200
        assert result.json()["items"] == []


def test_empty_experiments_paginate_without_losing_snapshots(
    persistent_api: PersistentAPI,
) -> None:
    api = persistent_api
    snapshot_ids = [upload(api, suffix=f"-empty-{index}") for index in range(3)]
    found = []
    for offset in range(3):
        result = api.client.get(
            "/api/v1/experiments",
            params={"limit": 1, "offset": offset},
            headers=api.headers(),
        )
        assert result.status_code == 200
        found.append(result.json()["items"][0]["snapshot_id"])
    assert found == [str(value) for value in reversed(snapshot_ids)]


def test_empty_experiments_preserve_delegated_subject_and_run_scope(
    context: IntegrationContext,
) -> None:
    snapshot_id = delegated_upload(context)
    for headers, expected in (
        (context.headers(), [str(snapshot_id)]),
        (context.headers("bob"), []),
        (context.headers(run="another-run"), []),
    ):
        result = context.client.get(PREFIX + "/experiments", headers=headers)
        assert result.status_code == 200
        assert [item["snapshot_id"] for item in result.json()["items"]] == expected
