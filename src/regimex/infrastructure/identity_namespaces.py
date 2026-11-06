"""Keep local accounts, integration accounts and delegated owners disjoint."""

from uuid import uuid4

import psycopg

from regimex.service_contracts import CapacityExceeded, Conflict

Connection = psycopg.Connection[tuple[object, ...]]


def _lock_owner(connection: Connection, owner_id: str) -> None:
    connection.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
        ("identity-owner:" + owner_id,),
    )


def _has_resources(connection: Connection, owner_id: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM ("
            "SELECT owner_id FROM regimex_snapshots WHERE owner_id=%s UNION ALL "
            "SELECT owner_id FROM regimex_operations WHERE owner_id=%s UNION ALL "
            "SELECT owner_id FROM regimex_datasets WHERE owner_id=%s UNION ALL "
            "SELECT owner_id FROM regimex_reference_sets WHERE owner_id=%s UNION ALL "
            "SELECT owner_id FROM regimex_publication_audit WHERE owner_id=%s UNION ALL "
            "SELECT owner_id FROM regimex_orphan_plans WHERE owner_id=%s"
            ") retained LIMIT 1",
            (owner_id,) * 6,
        ).fetchone()
        is not None
    )


def reserve_integration_owner(connection: Connection, owner_id: str) -> None:
    """Existing integration accounts may serve several explicitly scoped sources."""
    _lock_owner(connection, owner_id)
    collision = connection.execute(
        "SELECT 1 FROM regimex_tokens WHERE owner_id=%s AND role<>'integration' "
        "UNION ALL SELECT 1 FROM regimex_integration_subjects WHERE effective_owner_id=%s LIMIT 1",
        (owner_id, owner_id),
    ).fetchone()
    if collision is not None:
        raise Conflict("This account identity is already reserved for a local or delegated user.")
    existing = connection.execute(
        "SELECT 1 FROM regimex_integrations WHERE owner_id=%s LIMIT 1",
        (owner_id,),
    ).fetchone()
    if existing is None and _has_resources(connection, owner_id):
        raise Conflict("This account identity already owns retained resources.")


def reserve_local_owner(connection: Connection, owner_id: str) -> None:
    _lock_owner(connection, owner_id)
    if (
        connection.execute(
            "SELECT 1 FROM regimex_integrations WHERE owner_id=%s UNION ALL "
            "SELECT 1 FROM regimex_integration_subjects WHERE effective_owner_id=%s LIMIT 1",
            (owner_id, owner_id),
        ).fetchone()
        is not None
    ):
        raise Conflict("This account identity is reserved for an integration or delegated user.")


def new_subject_owner(connection: Connection) -> str:
    """Reserve a fresh opaque owner before persisting its immutable subject binding."""
    for _attempt in range(8):
        owner_id = "external_" + uuid4().hex
        _lock_owner(connection, owner_id)
        collision = connection.execute(
            "SELECT 1 FROM regimex_tokens WHERE owner_id=%s UNION ALL "
            "SELECT 1 FROM regimex_integrations WHERE owner_id=%s UNION ALL "
            "SELECT 1 FROM regimex_integration_subjects WHERE effective_owner_id=%s LIMIT 1",
            (owner_id, owner_id, owner_id),
        ).fetchone()
        if collision is None and not _has_resources(connection, owner_id):
            return owner_id
    raise CapacityExceeded("A distinct delegated account identity could not be allocated.")
