"""Reviewed GC against PostgreSQL and both actual storage implementations."""

import os
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Never, cast
from uuid import uuid4

import pytest

pytest.importorskip("psycopg", reason="install the service extra")
pytest.importorskip("alembic", reason="install the service extra")
pytest.importorskip("procrastinate", reason="install the service extra")

import psycopg
from test_object_store_contract import Backend, RawClient, digest, source
from test_object_store_contract import backend as backend
from test_retention import stopped

from regimex.application.retention import RetentionLimits, RetentionService
from regimex.infrastructure.database import DatabaseRepository
from regimex.infrastructure.object_store import ObjectInfo
from regimex.infrastructure.retention_repository import RetentionRepository
from regimex.infrastructure.s3_object_store import S3ObjectStore
from regimex.service_contracts import Principal


@pytest.fixture
def database() -> Iterator[DatabaseRepository]:
    dsn = os.environ.get("REGIMEX_TEST_DATABASE_DSN")
    if not dsn:
        pytest.skip("set REGIMEX_TEST_DATABASE_DSN to a disposable PostgreSQL database")
    repository = DatabaseRepository(dsn, schema="regimex_storage_gc_" + uuid4().hex)
    try:
        repository.migrate()
        yield repository
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(
                psycopg.sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    psycopg.sql.Identifier(repository.schema)
                )
            )


def test_same_review_fence_crash_and_new_generation_contract(
    database: DatabaseRepository,
    backend: Backend,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects = backend.create(1024, 4096)
    upload = source(tmp_path)
    registered = objects.put("repro/old-model", upload, expected_sha256=digest(b"payload"))
    orphan = objects.put("cancelled/attempt", upload, expected_sha256=registered.sha256)
    with database.connection() as connection:
        database._object(connection, registered)
    identity = Principal(owner_id="operator", role="admin", token_id=uuid4())
    service = RetentionService(
        RetentionRepository(database),
        objects,
        clock=lambda: datetime.now(UTC) + timedelta(days=2),
    )
    plan = service.plan(identity, attestation=stopped())
    assert [item.object for item in plan.objects] == [orphan]
    original = objects.delete_version

    def crash(info: ObjectInfo) -> Never:
        original(info)
        raise RuntimeError("artificial collector death after physical deletion")

    monkeypatch.setattr(objects, "delete_version", crash)
    with pytest.raises(RuntimeError, match="collector death"):
        service.apply(identity, plan.plan_id, expected_sha256=plan.sha256, attestation=stopped())
    with database.connection() as connection, pytest.raises(psycopg.errors.RaiseException):
        database._object(connection, orphan)
    monkeypatch.setattr(objects, "delete_version", original)
    receipt = service.apply(
        identity, plan.plan_id, expected_sha256=plan.sha256, attestation=stopped()
    )
    assert receipt.complete and receipt.events[-1].status == "already_missing"
    assert objects.stat(registered.key) == registered
    replacement = objects.put(orphan.key, upload, expected_sha256=orphan.sha256)
    assert replacement.version_id != orphan.version_id
    with database.connection() as connection:
        database._object(connection, replacement)
    assert objects.stat(replacement.key) == replacement


def test_owned_stopped_multipart_uploads_abort_after_review_and_grace(
    database: DatabaseRepository, backend: Backend
) -> None:
    objects = backend.create(1024, 4096)
    if not isinstance(objects, S3ObjectStore):
        assert objects.list_multipart_uploads() == ()
        return
    client = cast(RawClient, objects._sdk().client)
    client.create_multipart_upload(
        Bucket=objects.config.bucket, Key=objects.config.prefix + "/abandoned/incomplete"
    )
    time.sleep(1.1)
    identity = Principal(owner_id="operator", role="admin", token_id=uuid4())
    service = RetentionService(
        RetentionRepository(database), objects, limits=RetentionLimits(minimum_grace_seconds=1)
    )
    plan = service.plan(identity, attestation=stopped(), grace_seconds=1)
    assert len(plan.multipart) == 1 and not plan.objects
    receipt = service.apply(
        identity, plan.plan_id, expected_sha256=plan.sha256, attestation=stopped()
    )
    assert receipt.complete and receipt.events[-1].status == "mpu_aborted"
    assert not objects.list_multipart_uploads() and not objects.inventory()
