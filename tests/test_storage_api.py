"""Storage usage reads preserve owner, integration and capability boundaries."""

from uuid import uuid4

import pytest

pytest.importorskip("psycopg", reason="install the service extra")
pytest.importorskip("procrastinate", reason="install the queue-test group")
pytest.importorskip("httpx2", reason="install the service-test group")

from test_integration_scope import PREFIX, IntegrationContext, upload
from test_integration_scope import context as context

from regimex.access_contracts import IntegrationKeyCreate
from regimex.infrastructure.integration_repository import IntegrationRepository


def test_storage_usage_belongs_to_authenticated_owner(context: IntegrationContext) -> None:
    assert context.client.get("/api/v1/storage").status_code == 401
    snapshot_id = upload(context)
    saved = context.database.get_snapshot_internal(snapshot_id)
    response = context.client.get(PREFIX + "/storage", headers=context.headers())
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "private, no-store"
    assert response.json() == {
        "used_bytes": saved.object.size_bytes,
        "reserved_bytes": 0,
        "limit_bytes": 1024**3,
        "available_bytes": 1024**3 - saved.object.size_bytes,
    }
    assert "owner_id" not in response.json()
    assert "object_key" not in response.json()
    bob = context.client.get(PREFIX + "/storage", headers=context.headers("bob"))
    assert bob.status_code == 200
    assert bob.json()["used_bytes"] == 0
    user = context.client.get(
        "/api/v1/storage?owner_id=" + saved.owner_id,
        headers={"Authorization": "Bearer " + context.user_token},
    )
    assert user.status_code == 200
    assert user.json()["used_bytes"] == 0
    # Replaying the same snapshot does not consume additional space.
    assert upload(context) == snapshot_id
    repeated = context.client.get(PREFIX + "/storage", headers=context.headers())
    assert repeated.json() == response.json()


def test_connection_storage_is_an_administrator_aggregate(context: IntegrationContext) -> None:
    snapshot_id = upload(context)
    saved = context.database.get_snapshot_internal(snapshot_id)
    integration = context.database.authenticate(context.gateway_token)
    administrator_token = context.database.create_token("storage-administrator", "admin")
    path = f"/api/v1/admin/integrations/{integration.integration_id}/storage"
    assert context.client.get(path).status_code == 401
    assert (
        context.client.get(
            path, headers={"Authorization": "Bearer " + context.user_token}
        ).status_code
        == 403
    )
    administrator_headers = {"Authorization": "Bearer " + administrator_token}
    response = context.client.get(path, headers=administrator_headers)
    assert response.status_code == 200, response.text
    assert response.json() == {
        "used_bytes": saved.object.size_bytes,
        "reserved_bytes": 0,
        "limit_bytes": 2 * 1024**3,
        "available_bytes": 2 * 1024**3 - saved.object.size_bytes,
    }
    absent = context.client.get(
        f"/api/v1/admin/integrations/{uuid4()}/storage", headers=administrator_headers
    )
    assert absent.status_code == 404


def test_delegated_storage_usage_requires_read_capability(context: IntegrationContext) -> None:
    administrator = context.database.authenticate(
        context.database.create_token("storage-administrator", "admin")
    )
    integration = context.database.authenticate(context.gateway_token)
    assert integration.integration_id is not None
    repository = IntegrationRepository(context.database)
    for scopes, expected in ((("read",), 200), (("upload", "analyze"), 403)):
        key = repository.issue_key(
            administrator,
            integration.integration_id,
            IntegrationKeyCreate.model_validate({"name": "Storage reader", "scopes": scopes}),
        )
        response = context.client.get(
            PREFIX + "/storage", headers=context.headers(token=key.token.get_secret_value())
        )
        assert response.status_code == expected, response.text
