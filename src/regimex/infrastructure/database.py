"""PostgreSQL authority for access, publication and transactional queue effects.

Each method opens a short transaction. Computation and object transfers occur
outside these transactions; only the attempt holding the locked current token
can make their result visible. The repository never deletes stored objects.
"""

import hashlib
import re
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast
from uuid import UUID, uuid4

import procrastinate
import psycopg
from psycopg.types.json import Json, Jsonb
from pydantic import JsonValue, TypeAdapter, ValidationError

from regimex.contracts import MAX_INPUT_BYTES
from regimex.infrastructure.heartbeat_connection import HeartbeatConnection
from regimex.infrastructure.history_repository import analysis_filters, analysis_scope, search_text
from regimex.infrastructure.object_store import ObjectInfo
from regimex.operations import (
    OPERATION_VERSION,
    LaunchSource,
    Operation,
    OperationFailure,
    OperationKind,
    ResolvedModel,
    acknowledge_cancel,
    fail,
    heartbeat,
    recover_stale,
    request_cancel,
    start_attempt,
    succeed,
)
from regimex.report_contracts import MAX_REPORT_BYTES, SAVED_REPORT_RENDERERS, AnalysisReportRecord
from regimex.service_contracts import (
    AnalysisHistoryFilter,
    AnalysisRecord,
    AnalysisSummary,
    CapacityExceeded,
    Conflict,
    Forbidden,
    HistoryState,
    ModelRecord,
    NotFound,
    Principal,
    Role,
    ServiceError,
    SnapshotRecord,
    StaleAttempt,
    UnsupportedContract,
)
from regimex.storage import json_bytes

_RESULT = TypeAdapter(dict[str, JsonValue])
_REVISION = "0015_terminal_activity"


def _read_operation(document: object) -> Operation:
    if isinstance(document, dict) and document.get("version") != OPERATION_VERSION:
        raise UnsupportedContract()
    return Operation.model_validate_json(json_bytes(document))


def _schema_name(schema: str) -> str:
    if re.fullmatch(r"[a-z][a-z0-9_]{0,62}", schema) is None:
        raise ValueError("database schema must be a lowercase PostgreSQL identifier")
    return schema


def create_queue_app(dsn: str, *, schema: str = "public") -> procrastinate.App:
    return procrastinate.App(
        connector=procrastinate.SyncPsycopgConnector(
            conninfo=dsn,
            kwargs={"options": _connection_options(_schema_name(schema)), "connect_timeout": 5},
        )
    )


def _connection_options(schema: str) -> str:
    return (
        f"-csearch_path={schema} -cstatement_timeout=10000 "
        "-cidle_in_transaction_session_timeout=10000 -clock_timeout=5000"
    )


class DatabaseRepository:
    def __init__(
        self,
        dsn: str,
        *,
        schema: str = "public",
        queue_app: procrastinate.App | None = None,
    ) -> None:
        self._dsn = dsn
        self.schema = _schema_name(schema)
        self.queue_app = queue_app or create_queue_app(dsn, schema=schema)

    @contextmanager
    def connection(
        self, *, heartbeat: bool = False
    ) -> Iterator[psycopg.Connection[tuple[object, ...]]]:
        connect = HeartbeatConnection.connect if heartbeat else psycopg.connect
        with connect(
            self._dsn,
            options=_connection_options(self.schema),
            application_name="regimex",
            connect_timeout=5,
        ) as connection:
            # Check before opening the caller's transaction so maintenance can
            # select its own isolation level. Restores never overwrite a live DB.
            connection.autocommit = True
            guard = connection.execute("SELECT to_regclass('regimex_restore_state')").fetchone()
            if guard is None or guard[0] is None:
                raise ServiceError(
                    "schema_unavailable", "Apply service migrations before serving requests.", 503
                )
            state = connection.execute(
                "SELECT ready FROM regimex_restore_state WHERE singleton"
            ).fetchone()
            if state != (True,):
                raise ServiceError(
                    "restore_incomplete", "Offline restoration has not completed.", 503
                )
            connection.autocommit = False
            yield connection

    def migrate(self) -> None:
        """Explicit operator action; API startup never creates or changes schema."""
        from alembic import command
        from alembic.config import Config
        from sqlalchemy import create_engine

        engine = create_engine(
            "postgresql+psycopg://",
            creator=lambda: psycopg.connect(
                self._dsn, options=_connection_options(self.schema), connect_timeout=5
            ),
        )
        try:
            with engine.begin() as connection:
                connection.exec_driver_sql(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"regimex-migration:{self.schema}",),
                )
                connection.exec_driver_sql(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')
                config = Config()
                config.set_main_option("script_location", str(Path(__file__).parent / "migrations"))
                config.attributes["connection"] = connection
                config.attributes["schema"] = self.schema
                if (
                    connection.exec_driver_sql("SELECT to_regclass('alembic_version')").scalar_one()
                    is None
                ):
                    # Later application migrations may reference queue tables.
                    command.upgrade(config, "0001_initial")
                exists: object = connection.exec_driver_sql(
                    "SELECT to_regclass('procrastinate_jobs')"
                ).scalar_one()
                if exists is None:
                    # Execute the exact pinned queue library's schema on this same
                    # connection, including its own schema version metadata.
                    connection.exec_driver_sql(
                        self.queue_app.schema_manager.get_schema().replace("%", "%%")
                    )
                    connection.exec_driver_sql(
                        "INSERT INTO regimex_queue_schema (singleton,schema_sha256) "
                        "VALUES (true,%s)",
                        (self._queue_schema_sha256(),),
                    )
                else:
                    recorded = connection.exec_driver_sql(
                        "SELECT schema_sha256 FROM regimex_queue_schema WHERE singleton"
                    ).scalar_one_or_none()
                    if recorded != self._queue_schema_sha256():
                        raise RuntimeError("Queue schema differs from the pinned reviewed schema.")
                command.upgrade(config, "head")
        finally:
            engine.dispose()

    def readiness(self, *, require_model: bool = False) -> bool:
        try:
            with self.connection() as connection:
                row = connection.execute(
                    "SELECT to_regclass('alembic_version'), to_regclass('regimex_operations'), "
                    "to_regclass('procrastinate_jobs')"
                ).fetchone()
                if row is None or any(value is None for value in row):
                    return False
                revision = connection.execute("SELECT version_num FROM alembic_version").fetchone()
                queue_schema = connection.execute(
                    "SELECT schema_sha256 FROM regimex_queue_schema WHERE singleton"
                ).fetchone()
                if revision != (_REVISION,) or queue_schema != (self._queue_schema_sha256(),):
                    return False
                if require_model:
                    self._model(connection, "active")
                return True
        except psycopg.Error, ServiceError, ValidationError:
            return False

    def _queue_schema_sha256(self) -> str:
        return hashlib.sha256(self.queue_app.schema_manager.get_schema().encode()).hexdigest()

    @staticmethod
    def _admin(principal: Principal) -> None:
        if principal.role != "admin":
            raise Forbidden()

    def create_token(
        self, owner_id: str, role: Role, *, source_namespace: str | None = None
    ) -> str:
        """Provision locally; the raw credential is returned once and never persisted."""
        token_id = uuid4()
        principal = Principal(
            owner_id=owner_id,
            role=role,
            token_id=token_id,
            source_namespace=source_namespace,
        )
        if (role == "integration") != (source_namespace is not None):
            raise ValueError("only integration credentials require a bound source namespace")
        if role == "integration":
            from regimex.infrastructure.integration_repository import provision_integration_token

            assert source_namespace is not None
            return provision_integration_token(self, owner_id, source_namespace)
        token = "rgx_" + secrets.token_urlsafe(32)
        with self.connection() as connection:
            from regimex.infrastructure.identity_namespaces import reserve_local_owner

            reserve_local_owner(connection, owner_id)
            connection.execute(
                "INSERT INTO regimex_tokens "
                "(token_id,token_sha256,owner_id,role,source_namespace,created_at) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                (
                    token_id,
                    hashlib.sha256(token.encode()).hexdigest(),
                    principal.owner_id,
                    role,
                    source_namespace,
                    datetime.now(UTC),
                ),
            )
        return token

    def authenticate(self, token: str) -> Principal:
        from regimex.infrastructure.integration_repository import authenticate_token

        with self.connection() as connection:
            return authenticate_token(connection, token)

    def revoke_token(self, token_id: UUID) -> None:
        from regimex.infrastructure.integration_repository import revoke_token

        revoke_token(self, token_id)

    def issue_token(
        self, owner_id: str, role: Role, *, source_namespace: str | None = None
    ) -> tuple[Principal, str]:
        token = self.create_token(owner_id, role, source_namespace=source_namespace)
        return self.authenticate(token), token

    @staticmethod
    def _object(connection: psycopg.Connection[tuple[object, ...]], value: ObjectInfo) -> None:
        from regimex.infrastructure.resource_lifecycle import require_object_key

        require_object_key(connection, value.key)
        connection.execute(
            "INSERT INTO regimex_objects (object_key,sha256,size_bytes,version_id,document) "
            "VALUES (%s,%s,%s,%s,%s) ON CONFLICT (object_key) DO NOTHING",
            (
                value.key,
                value.sha256,
                value.size_bytes,
                value.version_id,
                Jsonb(value.model_dump()),
            ),
        )
        row = connection.execute(
            "SELECT document FROM regimex_objects WHERE object_key=%s", (value.key,)
        ).fetchone()
        if row is None or ObjectInfo.model_validate(row[0]) != value:
            raise Conflict(
                "The immutable object key already identifies different bytes or a version."
            )

    def add_snapshot(self, record: SnapshotRecord) -> SnapshotRecord:
        with self.connection() as connection:
            return self._add_snapshot(connection, record)

    @staticmethod
    def existing_snapshot(
        connection: psycopg.Connection[tuple[object, ...]],
        owner_id: str,
        source_id: str,
        run_id: str,
        source_snapshot_id: str,
        content_sha256: str,
    ) -> SnapshotRecord | None:
        from regimex.infrastructure.resource_lifecycle import require_snapshot_identity

        require_snapshot_identity(connection, owner_id, source_id, run_id, source_snapshot_id)
        row = connection.execute(
            "SELECT document FROM regimex_snapshots WHERE "
            "owner_id=%s AND source_id=%s AND run_id=%s AND source_snapshot_id=%s",
            (owner_id, source_id, run_id, source_snapshot_id),
        ).fetchone()
        if row is None:
            return None
        existing = SnapshotRecord.model_validate_json(json_bytes(row[0]))
        if existing.content_sha256 != content_sha256:
            raise Conflict("The source snapshot identity already has different content.")
        return existing

    def _add_snapshot(
        self, connection: psycopg.Connection[tuple[object, ...]], record: SnapshotRecord
    ) -> SnapshotRecord:
        from regimex.infrastructure.storage_quotas import record_owned_object

        self._lock(connection, f"owner-payload:{record.owner_id}")
        self._lock(connection, f"owner:{record.owner_id}")
        existing = self.existing_snapshot(
            connection,
            record.owner_id,
            record.source_id,
            record.run_id,
            record.source_snapshot_id,
            record.content_sha256,
        )
        if existing is not None:
            if existing.object.key == record.object.key and existing.object != record.object:
                raise Conflict("The immutable object key already identifies another generation.")
            return existing
        record_owned_object(connection, record.owner_id, record.source_id, record.object)
        self._object(connection, record.object)
        connection.execute(
            "INSERT INTO regimex_snapshots "
            "(snapshot_id,owner_id,source_id,run_id,source_snapshot_id,content_sha256,"
            "object_key,document,identifiers_search) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                record.snapshot_id,
                record.owner_id,
                record.source_id,
                record.run_id,
                record.source_snapshot_id,
                record.content_sha256,
                record.object.key,
                Jsonb(record.model_dump(mode="json")),
                search_text(
                    record.source_id,
                    record.run_id,
                    record.source_snapshot_id,
                    *record.instrument_ids,
                ),
            ),
        )
        return record

    def get_snapshot_internal(self, snapshot_id: UUID) -> SnapshotRecord:
        with self.connection() as connection:
            return self._snapshot(connection, snapshot_id)

    @staticmethod
    def _snapshot(
        connection: psycopg.Connection[tuple[object, ...]], snapshot_id: UUID
    ) -> SnapshotRecord:
        row = connection.execute(
            "SELECT document FROM regimex_snapshots WHERE snapshot_id=%s", (snapshot_id,)
        ).fetchone()
        if row is None:
            raise NotFound()
        return SnapshotRecord.model_validate_json(json_bytes(row[0]))

    def get_snapshot(self, principal: Principal, snapshot_id: UUID) -> SnapshotRecord:
        result = self.get_snapshot_internal(snapshot_id)
        self._authorize_snapshot(principal, result)
        return result

    def get_snapshots(
        self, principal: Principal, snapshot_ids: tuple[UUID, ...]
    ) -> tuple[SnapshotRecord, ...]:
        if not snapshot_ids:
            return ()
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT s.document FROM unnest(%s::uuid[]) WITH ORDINALITY AS requested(id, pos) "
                "JOIN regimex_snapshots s ON s.snapshot_id=requested.id AND s.owner_id=%s "
                "ORDER BY requested.pos",
                (list(snapshot_ids), principal.owner_id),
            ).fetchall()
        if len(rows) != len(snapshot_ids):
            raise NotFound()
        records = tuple(SnapshotRecord.model_validate_json(json_bytes(row[0])) for row in rows)
        for requested, record in zip(snapshot_ids, records, strict=True):
            self._authorize_snapshot(principal, record)
            if record.snapshot_id != requested:
                raise Conflict("Stored metadata differs from the requested snapshot identity.")
        return records

    @staticmethod
    def _authorize_snapshot(principal: Principal, snapshot: SnapshotRecord) -> None:
        if (
            snapshot.owner_id != principal.owner_id
            or (
                principal.role == "integration" and snapshot.source_id != principal.source_namespace
            )
            or (principal.run_scope is not None and snapshot.run_id != principal.run_scope)
        ):
            raise NotFound()

    def list_snapshots(
        self, principal: Principal, *, limit: int = 100, offset: int = 0
    ) -> tuple[SnapshotRecord, ...]:
        self._page(limit, offset)
        with self.connection() as connection:
            query = "SELECT document FROM regimex_snapshots WHERE owner_id=%s"
            parameters: list[object] = [principal.owner_id]
            if principal.role == "integration":
                query += " AND source_id=%s"
                parameters.append(principal.source_namespace)
                if principal.run_scope is not None:
                    query += " AND run_id=%s"
                    parameters.append(principal.run_scope)
            rows = connection.execute(
                query + " ORDER BY snapshot_id LIMIT %s OFFSET %s", [*parameters, limit, offset]
            ).fetchall()
        return tuple(SnapshotRecord.model_validate_json(json_bytes(row[0])) for row in rows)

    def register_model(self, principal: Principal, record: ModelRecord) -> ModelRecord:
        self._admin(principal)
        with self.connection() as connection:
            self.lock_catalog_identity(connection, "model", record.model_version_id)
            self._require_unreserved_version(connection, record.model_version_id)
            return self._register_model(connection, record)

    @staticmethod
    def _require_unreserved_version(
        connection: psycopg.Connection[tuple[object, ...]], model_version_id: str
    ) -> None:
        # A completed fit still owns its version after weight deletion. A stale
        # reservation alongside an independent import can replay that exact import.
        reserved = connection.execute(
            "SELECT 1 FROM regimex_training_runs t WHERE t.model_version_id=%s "
            "AND (t.committed_model_version_id IS NOT NULL OR NOT EXISTS "
            "(SELECT 1 FROM regimex_models m WHERE m.model_version_id=t.model_version_id))",
            (model_version_id,),
        ).fetchone()
        if reserved is not None:
            raise Conflict("This model version is reserved by an administrative training run.")

    def existing_registration(
        self, principal: Principal, model_version_id: str
    ) -> ModelRecord | None:
        self._admin(principal)
        with self.connection() as connection:
            try:
                return self._model(connection, model_version_id)
            except NotFound:
                # A pending or discarded candidate still owns its version ID.
                # Reject it before an import uploads another copy of the bundle.
                self._require_unreserved_version(connection, model_version_id)
                return None

    def _register_model(
        self, connection: psycopg.Connection[tuple[object, ...]], record: ModelRecord
    ) -> ModelRecord:
        if record.model_version_id == "active":
            raise Conflict("The active alias cannot be used as an immutable model version.")
        self._object(connection, record.manifest)
        for member in record.members.values():
            self._object(connection, member)
        connection.execute(
            "INSERT INTO regimex_models "
            "(model_version_id,architecture_id,bundle_sha256,manifest_key,document) "
            "VALUES (%s,%s,%s,%s,%s) ON CONFLICT (model_version_id) DO NOTHING",
            (
                record.model_version_id,
                record.architecture_id,
                record.bundle_sha256,
                record.manifest.key,
                Jsonb(record.model_dump(mode="json")),
            ),
        )
        existing = self._model(connection, record.model_version_id)
        if existing.model_dump(exclude={"created_at"}) != record.model_dump(exclude={"created_at"}):
            raise Conflict("The model version is immutable and already has other metadata.")
        for path, member in record.members.items():
            connection.execute(
                "INSERT INTO regimex_model_members (model_version_id,member_path,object_key) "
                "VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                (record.model_version_id, path, member.key),
            )
        return existing

    @staticmethod
    def _model(
        connection: psycopg.Connection[tuple[object, ...]],
        model_version_id: str,
        *,
        lock: bool = False,
    ) -> ModelRecord:
        if model_version_id == "active":
            row = connection.execute(
                "SELECT m.document FROM regimex_models m JOIN regimex_deployment d "
                "ON m.model_version_id=d.model_version_id WHERE d.singleton "
                "AND m.weights_deletion_id IS NULL"
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT document FROM regimex_models WHERE model_version_id=%s "
                "AND weights_deletion_id IS NULL" + (" FOR SHARE" if lock else ""),
                (model_version_id,),
            ).fetchone()
        if row is None:
            raise NotFound(
                "No registered model matches the requested version or active deployment."
            )
        return ModelRecord.model_validate_json(json_bytes(row[0]))

    def get_model(self, model_version_id: str) -> ModelRecord:
        with self.connection() as connection:
            return self._model(connection, model_version_id)

    def get_active_model(self) -> ModelRecord | None:
        try:
            return self.get_model("active")
        except NotFound:
            return None

    def list_models(
        self,
        principal: Principal,
        *,
        limit: int = 100,
        offset: int = 0,
        published_only: bool = False,
    ) -> tuple[ModelRecord, ...]:
        self._page(limit, offset)
        with self.connection() as connection:
            query = "SELECT document FROM regimex_models m WHERE weights_deletion_id IS NULL"
            if published_only or principal.role != "admin":
                query += (
                    " AND EXISTS (SELECT 1 FROM regimex_publication_audit p "
                    "WHERE p.model_version_id=m.model_version_id)"
                )
            rows = connection.execute(
                query + " ORDER BY model_version_id LIMIT %s OFFSET %s", (limit, offset)
            ).fetchall()
        return tuple(ModelRecord.model_validate_json(json_bytes(row[0])) for row in rows)

    @staticmethod
    def _published(
        connection: psycopg.Connection[tuple[object, ...]], model_version_id: str
    ) -> bool:
        return (
            connection.execute(
                "SELECT 1 FROM regimex_publication_audit WHERE model_version_id=%s LIMIT 1",
                (model_version_id,),
            ).fetchone()
            is not None
        )

    def get_model_for_principal(self, principal: Principal, model_version_id: str) -> ModelRecord:
        with self.connection() as connection:
            model = self._model(connection, model_version_id)
            if principal.role != "admin" and not self._published(
                connection, model.model_version_id
            ):
                raise NotFound()
            return model

    def get_visible_model(self, principal: Principal, model_version_id: str) -> ModelRecord:
        return self.get_model_for_principal(principal, model_version_id)

    def activate_model(
        self, principal: Principal, model_version_id: str, *, expected_previous: str | None
    ) -> ModelRecord:
        self._admin(principal)
        with self.connection() as connection:
            candidate = connection.execute(
                "SELECT 1 FROM regimex_training_runs WHERE model_version_id=%s AND NOT external",
                (model_version_id,),
            ).fetchone()
            if candidate is not None:
                raise Conflict("Training candidates require evaluated administrative publication.")
            return self._activate_model(
                connection,
                principal.owner_id,
                model_version_id,
                expected_previous=expected_previous,
            )

    def _activate_model(
        self,
        connection: psycopg.Connection[tuple[object, ...]],
        owner_id: str,
        model_version_id: str,
        *,
        expected_previous: str | None,
        evaluation_run_id: UUID | None = None,
        operation_id: UUID | None = None,
    ) -> ModelRecord:
        self._lock(connection, "publication")
        model = self._model(connection, model_version_id)
        row = connection.execute(
            "SELECT model_version_id FROM regimex_deployment WHERE singleton FOR UPDATE"
        ).fetchone()
        previous = None if row is None else cast(str, row[0])
        if previous != expected_previous:
            raise Conflict("The active deployment changed; reload before publishing.")
        connection.execute(
            "INSERT INTO regimex_deployment (singleton,model_version_id) VALUES (true,%s) "
            "ON CONFLICT (singleton) DO UPDATE SET model_version_id=EXCLUDED.model_version_id",
            (model.model_version_id,),
        )
        connection.execute(
            "INSERT INTO regimex_publication_audit "
            "(publication_id,owner_id,previous_model_version_id,model_version_id,published_at,"
            "evaluation_run_id,operation_id) VALUES (%s,%s,%s,%s,clock_timestamp(),%s,%s)",
            (uuid4(), owner_id, previous, model.model_version_id, evaluation_run_id, operation_id),
        )
        return model

    def _lock(self, connection: psycopg.Connection[tuple[object, ...]], scope: str) -> None:
        connection.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (f"regimex:{self.schema}:{scope}",),
        )

    def lock_catalog_identity(
        self,
        connection: psycopg.Connection[tuple[object, ...]],
        kind: Literal["dataset", "reference", "model"],
        identifier: str,
    ) -> None:
        # Job reservations and imported resources occupy separate tables. Both
        # writers must lock their common identity before checking the other table.
        self._lock(connection, f"catalog:{kind}:{identifier}")

    def _enqueue(
        self, connection: psycopg.Connection[tuple[object, ...]], operation: Operation
    ) -> None:
        job_id = self.queue_app.configure_task(
            name="regimex.analysis" if operation.kind == "analysis" else "regimex.admin",
            queue="analysis" if operation.kind == "analysis" else "admin",
            connection=connection,
            schedule_in=None
            if operation.attempt == 0
            else {"seconds": min(2**operation.attempt, 60)},
        ).defer(operation_id=str(operation.operation_id))
        connection.execute(
            "INSERT INTO regimex_deliveries (operation_id,job_id) VALUES (%s,%s)",
            (operation.operation_id, job_id),
        )

    def _retry_predecessor(
        self,
        connection: psycopg.Connection[tuple[object, ...]],
        principal: Principal,
        snapshot_id: UUID,
        retry_of: UUID,
    ) -> Operation:
        previous = self._summary(connection, principal, retry_of)
        if previous.snapshot_id != snapshot_id:
            raise NotFound()
        if previous.operation.state not in {"failed", "cancelled"}:
            raise ServiceError(
                "retry_not_available", "Only a failed or cancelled analysis can be retried.", 409
            )
        return previous.operation

    def submit_analysis(
        self,
        principal: Principal,
        snapshot_id: UUID,
        *,
        model_version_id: str = "active",
        idempotency_key: str,
        launch_source: LaunchSource | None = None,
        request_sha256: str,
        max_pending: int = 8,
        integration_max_pending: int = 32,
        retry_of: UUID | None = None,
    ) -> AnalysisRecord:
        if max_pending < 1 or not 1 <= integration_max_pending <= 1000:
            raise ValueError("pending quotas must be positive and integration quota at most 1000")
        with self.connection() as connection:
            if principal.role == "integration":
                from regimex.infrastructure.integration_limits import lock_integration
                from regimex.infrastructure.integration_repository import require_integration

                lock_integration(self, connection, principal)
                require_integration(connection, principal, scopes=("analyze",))
            # One owner lock protects the shared quota across keys. Check for a
            # replay before resolving "active" so it returns the original model.
            self._lock(connection, f"owner:{principal.owner_id}")
            row = connection.execute(
                "SELECT o.operation_id,o.request_sha256,a.retry_of FROM regimex_operations o "
                "LEFT JOIN regimex_analyses a ON a.operation_id=o.operation_id "
                "WHERE o.owner_id=%s AND o.kind='analysis' AND o.idempotency_key=%s",
                (principal.owner_id, idempotency_key),
            ).fetchone()
            if row is not None:
                replay = self._analysis(connection, cast(UUID, row[0]))
                self._authorize_snapshot(principal, self._snapshot(connection, replay.snapshot_id))
                if row[1] == request_sha256:
                    return replay
                # A stored request can predate recording the retry's predecessor.
                # Preserve its digest and bind an eligible predecessor once under the owner lock.
                intake = {"snapshot_id": str(snapshot_id), "model_version_id": model_version_id}
                if (
                    replay.snapshot_id == snapshot_id
                    and hashlib.sha256(json_bytes(intake)).hexdigest() == row[1]
                ):
                    if retry_of is None or retry_of == row[2]:
                        return replay
                    if row[2] is None:
                        previous = self._retry_predecessor(
                            connection, principal, snapshot_id, retry_of
                        )
                        if (
                            previous.operation_id != replay.operation.operation_id
                            and previous.updated_at <= replay.operation.created_at
                        ):
                            connection.execute(
                                "UPDATE regimex_analyses SET retry_of=%s WHERE analysis_id=%s",
                                (retry_of, replay.analysis_id),
                            )
                            return replay
                raise Conflict("The idempotency key was already used for a different request.")
            snapshot = self._snapshot(connection, snapshot_id)
            self._authorize_snapshot(principal, snapshot)
            if retry_of is not None:
                self._retry_predecessor(connection, principal, snapshot_id, retry_of)
            count = connection.execute(
                "SELECT count(*) FROM regimex_operations WHERE owner_id=%s "
                "AND state IN ('queued','running','cancel_requested')",
                (principal.owner_id,),
            ).fetchone()
            assert count is not None
            if cast(int, count[0]) >= max_pending:
                raise CapacityExceeded()
            if principal.role == "integration":
                from regimex.infrastructure.integration_limits import check_pending

                check_pending(connection, principal, integration_max_pending)
            try:
                model = self._model(connection, model_version_id)
            except NotFound:
                if model_version_id != "active":
                    raise
                raise ServiceError(
                    "model_unavailable",
                    "Ask an administrator to activate a published model.",
                    409,
                ) from None
            if not self._published(connection, model.model_version_id):
                raise NotFound("Analysis requires an explicitly published model version.")
            operation_id = uuid4()
            now = datetime.now(UTC)
            operation = Operation(
                operation_id=operation_id,
                owner_id=principal.owner_id,
                kind="analysis",
                launch_source=launch_source,
                resource_id=str(operation_id),
                request_sha256=request_sha256,
                idempotency_key=idempotency_key,
                model=ResolvedModel(
                    model_version_id=model.model_version_id,
                    bundle_sha256=model.bundle_sha256,
                    inference_policy_sha256=model.inference_policy_sha256,
                ),
                created_at=now,
                updated_at=now,
            )
            connection.execute(
                "INSERT INTO regimex_operations "
                "(operation_id,owner_id,kind,idempotency_key,request_sha256,state,document) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (
                    operation_id,
                    operation.owner_id,
                    operation.kind,
                    idempotency_key,
                    request_sha256,
                    operation.state,
                    Jsonb(operation.model_dump(mode="json")),
                ),
            )
            connection.execute(
                "INSERT INTO regimex_analyses "
                "(analysis_id,operation_id,snapshot_id,model_version_id,"
                "model_version_search,retry_of) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                (
                    operation_id,
                    operation_id,
                    snapshot_id,
                    model.model_version_id,
                    search_text(model.model_version_id),
                    retry_of,
                ),
            )
            self._enqueue(connection, operation)
            return AnalysisRecord(
                analysis_id=operation_id, snapshot_id=snapshot_id, operation=operation
            )

    @staticmethod
    def _analysis(
        connection: psycopg.Connection[tuple[object, ...]], operation_id: UUID
    ) -> AnalysisRecord:
        row = connection.execute(
            "SELECT a.analysis_id,a.snapshot_id,o.document,a.result FROM regimex_analyses a "
            "JOIN regimex_operations o ON a.operation_id=o.operation_id WHERE o.operation_id=%s",
            (operation_id,),
        ).fetchone()
        if row is None:
            raise NotFound()
        record = AnalysisRecord(
            analysis_id=cast(UUID, row[0]),
            snapshot_id=cast(UUID, row[1]),
            operation=_read_operation(row[2]),
            result=None if row[3] is None else _RESULT.validate_python(row[3], strict=True),
        )
        if (record.operation.state == "succeeded") != (record.result is not None) or (
            record.result is not None
            and hashlib.sha256(json_bytes(record.result)).hexdigest()
            != record.operation.result_sha256
        ):
            raise Conflict("The persisted analysis result does not match its committed digest.")
        return record

    def get_analysis_internal(self, operation_id: UUID) -> AnalysisRecord:
        with self.connection() as connection:
            return self._analysis(connection, operation_id)

    def get_analysis(self, principal: Principal, analysis_id: UUID) -> AnalysisRecord:
        with self.connection() as connection:
            self._summary(connection, principal, analysis_id)
            return self._analysis(connection, analysis_id)

    @staticmethod
    def _summary_row(row: tuple[object, ...]) -> AnalysisSummary:
        snapshot = SnapshotRecord.model_validate_json(json_bytes(row[5]))
        summary = AnalysisSummary(
            analysis_id=cast(UUID, row[0]),
            snapshot_id=cast(UUID, row[1]),
            snapshot_sha256=snapshot.content_sha256,
            source_id=snapshot.source_id,
            run_id=snapshot.run_id,
            source_snapshot_id=snapshot.source_snapshot_id,
            instrument_ids=snapshot.instrument_ids,
            step_count=snapshot.step_count,
            operation=_read_operation(row[2]),
            result_available=cast(bool, row[3]),
            report_available=cast(bool, row[4]),
        )
        if summary.report_available and not summary.result_available:
            raise Conflict("The report is visible without a committed result.")
        if (summary.operation.state == "succeeded") != summary.result_available:
            raise Conflict("The analysis result visibility differs from its committed state.")
        return summary

    @staticmethod
    def _summary_query(principal: Principal) -> tuple[str, list[object]]:
        scope, parameters = analysis_scope(principal)
        return (
            "SELECT a.analysis_id,a.snapshot_id,o.document,a.result IS NOT NULL, "
            "EXISTS(SELECT 1 FROM regimex_analysis_reports r WHERE r.analysis_id=a.analysis_id), "
            "s.document " + scope,
            parameters,
        )

    @classmethod
    def _summary(
        cls,
        connection: psycopg.Connection[tuple[object, ...]],
        principal: Principal,
        analysis_id: UUID,
    ) -> AnalysisSummary:
        query, parameters = cls._summary_query(principal)
        row = connection.execute(
            query + " AND a.analysis_id=%s", [*parameters, analysis_id]
        ).fetchone()
        if row is None:
            raise NotFound()
        return cls._summary_row(row)

    def get_analysis_summary(self, principal: Principal, analysis_id: UUID) -> AnalysisSummary:
        with self.connection() as connection:
            return self._summary(connection, principal, analysis_id)

    def list_analysis_summaries(
        self,
        principal: Principal,
        *,
        limit: int = 100,
        offset: int = 0,
        q: str = "",
        state: HistoryState | None = None,
        snapshot_id: UUID | None = None,
    ) -> tuple[AnalysisSummary, ...]:
        self._page(limit, offset)
        predicate, filter_parameters = analysis_filters(AnalysisHistoryFilter(q=q, state=state))
        with self.connection() as connection:
            query, parameters = self._summary_query(principal)
            if snapshot_id is not None:
                query += " AND a.snapshot_id=%s"
                parameters.append(snapshot_id)
            query += " AND " + predicate
            parameters.extend(filter_parameters)
            rows = connection.execute(
                query + " ORDER BY (o.document->>'created_at')::timestamptz DESC,"
                "a.analysis_id DESC LIMIT %s OFFSET %s",
                [*parameters, limit, offset],
            ).fetchall()
            return tuple(self._summary_row(row) for row in rows)

    def list_analyses(
        self, principal: Principal, *, limit: int = 100, offset: int = 0
    ) -> tuple[AnalysisRecord, ...]:
        self._page(limit, offset)
        with self.connection() as connection:
            query = (
                "SELECT o.operation_id FROM regimex_operations o JOIN regimex_analyses a "
                "ON o.operation_id=a.operation_id JOIN regimex_snapshots s "
                "ON a.snapshot_id=s.snapshot_id WHERE o.owner_id=%s"
            )
            parameters: list[object] = [principal.owner_id]
            if principal.role == "integration":
                query += " AND s.source_id=%s"
                parameters.append(principal.source_namespace)
                if principal.run_scope is not None:
                    query += " AND s.run_id=%s"
                    parameters.append(principal.run_scope)
            rows = connection.execute(
                query + " ORDER BY (o.document->>'created_at')::timestamptz DESC,"
                "a.analysis_id DESC LIMIT %s OFFSET %s",
                [*parameters, limit, offset],
            ).fetchall()
            return tuple(self._analysis(connection, cast(UUID, row[0])) for row in rows)

    @staticmethod
    def _page(limit: int, offset: int) -> None:
        if not 1 <= limit <= 100 or not 0 <= offset <= 1_000_000:
            raise ValueError("pagination requires limit 1..100 and offset 0..1000000")

    @staticmethod
    def _operation(
        connection: psycopg.Connection[tuple[object, ...]],
        operation_id: UUID,
        *,
        lock: bool = False,
        principal: Principal | None = None,
    ) -> Operation:
        query = "SELECT o.document FROM regimex_operations o WHERE o.operation_id=%s"
        parameters: list[object] = [operation_id]
        if principal is not None:
            # Establish ownership before reporting an unsupported stored version.
            query += " AND o.owner_id=%s"
            parameters.append(principal.owner_id)
            query += (
                " AND NOT EXISTS(SELECT 1 FROM regimex_deleted_analyses d "
                "WHERE d.analysis_id=o.operation_id)"
            )
            if principal.role != "admin":
                query += " AND o.kind='analysis'"
            if principal.role == "integration":
                query += (
                    " AND EXISTS(SELECT 1 FROM regimex_analyses a JOIN regimex_snapshots s "
                    "ON a.snapshot_id=s.snapshot_id WHERE a.operation_id=o.operation_id "
                    "AND s.source_id=%s"
                )
                parameters.append(principal.source_namespace)
                if principal.run_scope is not None:
                    query += " AND s.run_id=%s"
                    parameters.append(principal.run_scope)
                query += ")"
        if lock:
            query += " FOR UPDATE OF o"
        row = connection.execute(query, parameters).fetchone()
        if row is None:
            raise NotFound()
        return _read_operation(row[0])

    @staticmethod
    def _save(connection: psycopg.Connection[tuple[object, ...]], operation: Operation) -> None:
        connection.execute(
            "UPDATE regimex_operations SET state=%s,heartbeat_at=%s,document=%s "
            "WHERE operation_id=%s",
            (
                operation.state,
                operation.heartbeat_at,
                Jsonb(operation.model_dump(mode="json")),
                operation.operation_id,
            ),
        )
        if operation.kind == "training" and operation.state not in {"running", "cancel_requested"}:
            # A retried fit starts a fresh curve. Successful runs keep their full
            # history in the model bundle; terminal attempts release preview space.
            connection.execute(
                "DELETE FROM regimex_training_progress WHERE operation_id=%s",
                (operation.operation_id,),
            )

    @staticmethod
    def _now(operation: Operation) -> datetime:
        return max(datetime.now(UTC), operation.updated_at)

    def get_operation(self, principal: Principal, operation_id: UUID) -> Operation:
        with self.connection() as connection:
            operation = self._operation(connection, operation_id, principal=principal)
            if operation.owner_id != principal.owner_id:
                raise NotFound()
            self._authorize_operation(connection, principal, operation)
            return operation

    def get_operation_internal(self, operation_id: UUID) -> Operation:
        """Workers may still unwind after an owner removed a terminal analysis."""
        with self.connection() as connection:
            return self._operation(connection, operation_id)

    def _authorize_operation(
        self,
        connection: psycopg.Connection[tuple[object, ...]],
        principal: Principal,
        operation: Operation,
    ) -> None:
        if operation.kind != "analysis" and principal.role != "admin":
            raise NotFound()
        if principal.role == "integration":
            self._summary(connection, principal, operation.operation_id)

    def cancel_operation(self, principal: Principal, operation_id: UUID) -> Operation:
        with self.connection() as connection:
            operation = self._operation(connection, operation_id, lock=True, principal=principal)
            if operation.owner_id != principal.owner_id:
                raise NotFound()
            self._authorize_operation(connection, principal, operation)
            operation = request_cancel(operation, self._now(operation))
            self._save(connection, operation)
            return operation

    def claim_operation(self, operation_id: UUID) -> Operation | None:
        with self.connection() as connection:
            operation = self._operation(connection, operation_id, lock=True)
            if operation.state != "queued":
                return None
            operation = start_attempt(operation, uuid4(), self._now(operation))
            self._save(connection, operation)
            return operation

    def heartbeat_operation(self, operation_id: UUID, token: UUID, stage: str) -> Operation:
        with self.connection(heartbeat=True) as connection:
            operation = self._operation(connection, operation_id, lock=True)
            try:
                operation = heartbeat(operation, token, stage, self._now(operation))
            except ValueError as exc:
                raise StaleAttempt() from exc
            self._save(connection, operation)
            return operation

    def get_analysis_report(
        self, principal: Principal, analysis_id: UUID
    ) -> AnalysisReportRecord | None:
        with self.connection() as connection:
            summary = self._summary(connection, principal, analysis_id)
            if not summary.report_available:
                return None
            row = connection.execute(
                "SELECT document FROM regimex_analysis_reports WHERE analysis_id=%s",
                (analysis_id,),
            ).fetchone()
            if row is None:
                raise Conflict("The report metadata is absent after its visibility check.")
            if (
                isinstance(row[0], dict)
                and row[0].get("renderer_version") not in SAVED_REPORT_RENDERERS
            ):
                raise UnsupportedContract()
            return AnalysisReportRecord.model_validate_json(json_bytes(row[0]))

    @contextmanager
    def completion_transaction(
        self, operation_id: UUID, token: UUID, kind: OperationKind
    ) -> Iterator[tuple[psycopg.Connection[tuple[object, ...]], Operation]]:
        """Lock the attempt and settle a pending cancellation before publication."""
        with self.connection() as connection:
            operation = self._operation(connection, operation_id, lock=True)
            if operation.kind != kind or operation.attempt_token != token:
                raise StaleAttempt()
            if operation.state == "running":
                yield connection, operation
                return
            if operation.state != "cancel_requested":
                raise StaleAttempt()
            self._save(connection, acknowledge_cancel(operation, token, self._now(operation)))
        # Commit cancellation before telling the worker to discard its output.
        raise StaleAttempt()

    def complete_analysis(
        self,
        operation_id: UUID,
        token: UUID,
        result: dict[str, JsonValue],
        *,
        report: AnalysisReportRecord,
    ) -> Operation:
        if type(report) is not AnalysisReportRecord:
            raise ValueError("Analysis completion requires current report metadata.")
        checked = _RESULT.validate_python(result, strict=True)
        payload = json_bytes(checked)
        if len(payload) > MAX_INPUT_BYTES:
            raise ValueError("analysis result exceeds the 16 MiB result limit")
        digest = hashlib.sha256(payload).hexdigest()
        with self.completion_transaction(operation_id, token, "analysis") as (
            connection,
            operation,
        ):
            completed = succeed(operation, token, digest, self._now(operation))
            report = AnalysisReportRecord.model_validate_json(report.model_dump_json())
            analysis = self._analysis(connection, operation_id)
            snapshot = self._snapshot(connection, analysis.snapshot_id)
            if (
                report.analysis_id != analysis.analysis_id
                or report.owner_id != operation.owner_id
                or report.snapshot_id != snapshot.snapshot_id
                or report.snapshot_sha256 != snapshot.content_sha256
                or report.result_sha256 != digest
                or report.model != operation.model
                or (report.source.source_id, report.source.run_id, report.source.snapshot_id)
                != (snapshot.source_id, snapshot.run_id, snapshot.source_snapshot_id)
                or checked.get("analysis_id") != str(report.analysis_id)
                or checked.get("snapshot_id") != str(report.snapshot_id)
                or checked.get("snapshot_sha256") != report.snapshot_sha256
                or checked.get("source") != report.source.model_dump(mode="json")
                or checked.get("model") != report.model.model_dump(mode="json")
                or report.object.size_bytes > MAX_REPORT_BYTES
            ):
                raise Conflict("The report identity differs from the fenced analysis result.")
            from regimex.infrastructure.storage_quotas import record_owned_object

            record_owned_object(connection, report.owner_id, snapshot.source_id, report.object)
            self._object(connection, report.object)
            connection.execute(
                "INSERT INTO regimex_analysis_reports "
                "(analysis_id,snapshot_id,owner_id,model_version_id,"
                "result_sha256,object_key,document) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (
                    report.analysis_id,
                    report.snapshot_id,
                    report.owner_id,
                    report.model.model_version_id,
                    report.result_sha256,
                    report.object.key,
                    Jsonb(report.model_dump(mode="json")),
                ),
            )
            connection.execute(
                "UPDATE regimex_analyses SET result=%s WHERE operation_id=%s",
                (Json(checked), operation_id),
            )
            self._save(connection, completed)
            return completed

    def fail_operation(
        self, operation_id: UUID, token: UUID, failure: OperationFailure
    ) -> Operation:
        with self.connection() as connection:
            operation = self._operation(connection, operation_id, lock=True)
            try:
                operation = fail(operation, token, failure, self._now(operation))
            except ValueError as exc:
                raise StaleAttempt() from exc
            self._save(connection, operation)
            if operation.state == "queued":
                self._enqueue(connection, operation)
            return operation

    def recover_operations(
        self, stale_before: datetime, *, limit: int = 100
    ) -> tuple[Operation, ...]:
        if not 1 <= limit <= 1000:
            raise ValueError("recovery limit must be 1..1000")
        if stale_before.tzinfo is None or stale_before.utcoffset() is None:
            raise ValueError("stale cutoff must be timezone aware")
        if stale_before > datetime.now(UTC):
            raise ValueError("stale cutoff must not be in the future")
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT document FROM regimex_operations WHERE "
                "state IN ('running','cancel_requested') AND heartbeat_at<=%s "
                "AND document->'version'=%s "
                "ORDER BY heartbeat_at LIMIT %s FOR UPDATE SKIP LOCKED",
                (stale_before, Jsonb(OPERATION_VERSION), limit),
            ).fetchall()
            recovered: list[Operation] = []
            for row in rows:
                operation = _read_operation(row[0])
                operation = recover_stale(operation, stale_before, self._now(operation))
                self._save(connection, operation)
                if operation.state == "queued":
                    self._enqueue(connection, operation)
                recovered.append(operation)
            return tuple(recovered)

    def recover_stale(self, stale_before: datetime) -> tuple[Operation, ...]:
        return self.recover_operations(stale_before)

    def prune_successful_recovery_deliveries(self) -> None:
        """Expire successful maintenance ticks, including any backlog from earlier runs."""
        with self.connection() as connection:
            # The queue removes the associated events and keeps its periodic
            # deduplication marker when the completed job is deleted.
            connection.execute(
                "WITH finished AS (SELECT id FROM procrastinate_jobs "
                "WHERE queue_name='maintenance' AND task_name='regimex.recover' "
                "AND status='succeeded' ORDER BY id LIMIT 1000 FOR UPDATE SKIP LOCKED) "
                "DELETE FROM procrastinate_jobs j USING finished WHERE j.id=finished.id"
            )

    def recover_queued_deliveries(
        self, *, limit: int = 100, stalled_worker_seconds: int | None = None
    ) -> tuple[Operation, ...]:
        """Bound recovery when delivery dies before the application can claim it.

        Scheduled retries count as live deliveries. A lost delivery consumes an
        application attempt before requeue, preventing unlimited repair loops.
        Delivery IDs remain when queue retention removes their corresponding jobs.
        """
        if not 1 <= limit <= 1000:
            raise ValueError("recovery limit must be 1..1000")
        if stalled_worker_seconds is not None and stalled_worker_seconds < 1:
            raise ValueError("worker timeout must be positive")
        with self.connection() as connection:
            if stalled_worker_seconds is not None:
                # A lost application delivery consumes an attempt below. A lost
                # periodic maintenance job is superseded by this recovery pass.
                connection.execute(
                    "WITH stalled AS MATERIALIZED (SELECT j.id FROM procrastinate_jobs j "
                    "LEFT JOIN procrastinate_workers w ON w.id=j.worker_id "
                    "WHERE j.status='doing' AND (w.id IS NULL OR "
                    "w.last_heartbeat < now()-make_interval(secs => %s)) AND "
                    "((j.task_name='regimex.recover' AND j.queue_name='maintenance') OR "
                    "EXISTS (SELECT 1 FROM regimex_deliveries d WHERE d.job_id=j.id)) "
                    "ORDER BY j.id LIMIT %s FOR UPDATE OF j SKIP LOCKED) "
                    "SELECT procrastinate_finish_job_v1(id,'failed',false) FROM stalled",
                    (stalled_worker_seconds, limit),
                )
            rows = connection.execute(
                "SELECT o.document FROM regimex_operations o WHERE o.state='queued' "
                "AND o.document->'version'=%s "
                "AND NOT EXISTS (SELECT 1 FROM regimex_deliveries d "
                "JOIN procrastinate_jobs j ON j.id=d.job_id "
                "WHERE d.operation_id=o.operation_id AND j.status IN ('todo','doing')) "
                "ORDER BY o.operation_id LIMIT %s FOR UPDATE OF o SKIP LOCKED",
                (Jsonb(OPERATION_VERSION), limit),
            ).fetchall()
            recovered: list[Operation] = []
            for row in rows:
                operation = _read_operation(row[0])
                token = uuid4()
                # A lost delivery consumes an attempt without any time in a worker.
                recovered_at = self._now(operation)
                operation = start_attempt(operation, token, recovered_at)
                operation = fail(
                    operation,
                    token,
                    OperationFailure(
                        code="infrastructure_failure",
                        message="The queue delivery ended before the operation could be claimed.",
                        retryable=True,
                    ),
                    recovered_at,
                )
                self._save(connection, operation)
                if operation.state == "queued":
                    self._enqueue(connection, operation)
                recovered.append(operation)
            return tuple(recovered)
