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
from typing import cast
from uuid import UUID, uuid4

import procrastinate
import psycopg
from alembic import command
from alembic.config import Config
from psycopg.types.json import Json, Jsonb
from pydantic import JsonValue, TypeAdapter
from sqlalchemy import create_engine

from regimex.contracts import MAX_INPUT_BYTES
from regimex.infrastructure.object_store import ObjectInfo
from regimex.operations import (
    OPERATION_VERSION,
    Operation,
    OperationFailure,
    ResolvedModel,
    fail,
    heartbeat,
    recover_stale,
    request_cancel,
    start_attempt,
    succeed,
)
from regimex.report_contracts import MAX_REPORT_BYTES, REPORT_RENDERER_VERSION, AnalysisReportRecord
from regimex.service_contracts import (
    AnalysisRecord,
    AnalysisSummary,
    CapacityExceeded,
    Conflict,
    Forbidden,
    ModelRecord,
    NotFound,
    Principal,
    Role,
    ServiceError,
    SnapshotRecord,
    StaleAttempt,
    Unauthorized,
    UnsupportedContract,
)
from regimex.storage import json_bytes

_RESULT = TypeAdapter(dict[str, JsonValue])
_REVISION = "0001_initial"


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
    def connection(self) -> Iterator[psycopg.Connection[tuple[object, ...]]]:
        with psycopg.connect(
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
                command.upgrade(config, "head")
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
        finally:
            engine.dispose()

    def readiness(self) -> bool:
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
                return revision == (_REVISION,) and queue_schema == (self._queue_schema_sha256(),)
        except psycopg.Error, ServiceError:
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
        token = "rgx_" + secrets.token_urlsafe(32)
        with self.connection() as connection:
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
        if not token.startswith("rgx_") or len(token) != 47 or not token.isascii():
            raise Unauthorized()
        with self.connection() as connection:
            row = connection.execute(
                "SELECT token_id,owner_id,role,source_namespace FROM regimex_tokens "
                "WHERE token_sha256=%s AND revoked_at IS NULL",
                (hashlib.sha256(token.encode()).hexdigest(),),
            ).fetchone()
        if row is None:
            raise Unauthorized()
        return Principal.model_validate(
            dict(zip(("token_id", "owner_id", "role", "source_namespace"), row, strict=True))
        )

    def revoke_token(self, token_id: UUID) -> None:
        with self.connection() as connection:
            row = connection.execute(
                "UPDATE regimex_tokens SET revoked_at=COALESCE(revoked_at,clock_timestamp()) "
                "WHERE token_id=%s RETURNING token_id",
                (token_id,),
            ).fetchone()
            if row is None:
                raise NotFound()

    def issue_token(
        self, owner_id: str, role: Role, *, source_namespace: str | None = None
    ) -> tuple[Principal, str]:
        token = self.create_token(owner_id, role, source_namespace=source_namespace)
        return self.authenticate(token), token

    @staticmethod
    def _object(connection: psycopg.Connection[tuple[object, ...]], value: ObjectInfo) -> None:
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

    def _add_snapshot(
        self, connection: psycopg.Connection[tuple[object, ...]], record: SnapshotRecord
    ) -> SnapshotRecord:
        self._object(connection, record.object)
        connection.execute(
            "INSERT INTO regimex_snapshots "
            "(snapshot_id,owner_id,source_id,run_id,source_snapshot_id,content_sha256,"
            "object_key,document) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT (owner_id,source_id,run_id,source_snapshot_id) DO NOTHING",
            (
                record.snapshot_id,
                record.owner_id,
                record.source_id,
                record.run_id,
                record.source_snapshot_id,
                record.content_sha256,
                record.object.key,
                Jsonb(record.model_dump(mode="json")),
            ),
        )
        row = connection.execute(
            "SELECT document FROM regimex_snapshots WHERE "
            "owner_id=%s AND source_id=%s AND run_id=%s AND source_snapshot_id=%s",
            (record.owner_id, record.source_id, record.run_id, record.source_snapshot_id),
        ).fetchone()
        assert row is not None
        existing = SnapshotRecord.model_validate_json(json_bytes(row[0]))
        if existing.content_sha256 != record.content_sha256:
            raise Conflict("The source snapshot identity already has different content.")
        return existing

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
            reserved = connection.execute(
                "SELECT 1 FROM regimex_training_runs WHERE model_version_id=%s",
                (record.model_version_id,),
            ).fetchone()
            if reserved is not None:
                raise Conflict("This model version is reserved by an administrative training run.")
            return self._register_model(connection, record)

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
        connection: psycopg.Connection[tuple[object, ...]], model_version_id: str
    ) -> ModelRecord:
        if model_version_id == "active":
            row = connection.execute(
                "SELECT m.document FROM regimex_models m JOIN regimex_deployment d "
                "ON m.model_version_id=d.model_version_id WHERE d.singleton"
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT document FROM regimex_models WHERE model_version_id=%s",
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
        self, principal: Principal, *, limit: int = 100, offset: int = 0
    ) -> tuple[ModelRecord, ...]:
        self._page(limit, offset)
        with self.connection() as connection:
            query = "SELECT document FROM regimex_models m"
            if principal.role != "admin":
                query += (
                    " WHERE EXISTS (SELECT 1 FROM regimex_publication_audit p "
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
                "SELECT 1 FROM regimex_training_runs WHERE model_version_id=%s",
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

    def submit_analysis(
        self,
        principal: Principal,
        snapshot_id: UUID,
        *,
        model_version_id: str = "active",
        idempotency_key: str,
        request_sha256: str,
        max_pending: int = 8,
    ) -> AnalysisRecord:
        if max_pending < 1:
            raise ValueError("pending quota must be positive")
        with self.connection() as connection:
            # Serialize every intake for this owner: different keys must not race
            # the pending quota, and a replay must precede active resolution.
            self._lock(connection, f"owner:{principal.owner_id}")
            row = connection.execute(
                "SELECT operation_id,request_sha256 FROM regimex_operations "
                "WHERE owner_id=%s AND kind='analysis' AND idempotency_key=%s",
                (principal.owner_id, idempotency_key),
            ).fetchone()
            if row is not None:
                replay = self._analysis(connection, cast(UUID, row[0]))
                self._authorize_snapshot(principal, self._snapshot(connection, replay.snapshot_id))
                if row[1] != request_sha256:
                    raise Conflict("The idempotency key was already used for a different request.")
                return replay
            snapshot = self._snapshot(connection, snapshot_id)
            self._authorize_snapshot(principal, snapshot)
            count = connection.execute(
                "SELECT count(*) FROM regimex_operations WHERE owner_id=%s "
                "AND state IN ('queued','running','cancel_requested')",
                (principal.owner_id,),
            ).fetchone()
            assert count is not None
            if cast(int, count[0]) >= max_pending:
                raise CapacityExceeded()
            model = self._model(connection, model_version_id)
            if not self._published(connection, model.model_version_id):
                raise NotFound("Analysis requires an explicitly published model version.")
            operation_id = uuid4()
            now = datetime.now(UTC)
            operation = Operation(
                operation_id=operation_id,
                owner_id=principal.owner_id,
                kind="analysis",
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
                "(analysis_id,operation_id,snapshot_id,model_version_id) VALUES (%s,%s,%s,%s)",
                (operation_id, operation_id, snapshot_id, model.model_version_id),
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
        summary = AnalysisSummary(
            analysis_id=cast(UUID, row[0]),
            snapshot_id=cast(UUID, row[1]),
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
        query = (
            "SELECT a.analysis_id,a.snapshot_id,o.document,a.result IS NOT NULL, "
            "EXISTS(SELECT 1 FROM regimex_analysis_reports r WHERE r.analysis_id=a.analysis_id) "
            "FROM regimex_analyses a JOIN regimex_operations o "
            "ON a.operation_id=o.operation_id JOIN regimex_snapshots s "
            "ON a.snapshot_id=s.snapshot_id WHERE o.owner_id=%s"
        )
        parameters: list[object] = [principal.owner_id]
        if principal.role == "integration":
            query += " AND s.source_id=%s"
            parameters.append(principal.source_namespace)
            if principal.run_scope is not None:
                query += " AND s.run_id=%s"
                parameters.append(principal.run_scope)
        return query, parameters

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
        self, principal: Principal, *, limit: int = 100, offset: int = 0
    ) -> tuple[AnalysisSummary, ...]:
        self._page(limit, offset)
        with self.connection() as connection:
            query, parameters = self._summary_query(principal)
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
        with self.connection() as connection:
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
                and row[0].get("renderer_version") != REPORT_RENDERER_VERSION
            ):
                raise UnsupportedContract()
            return AnalysisReportRecord.model_validate_json(json_bytes(row[0]))

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
        with self.connection() as connection:
            operation = self._operation(connection, operation_id, lock=True)
            try:
                completed = succeed(operation, token, digest, self._now(operation))
            except ValueError as exc:
                raise StaleAttempt() from exc
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
                "ORDER BY heartbeat_at LIMIT %s FOR UPDATE SKIP LOCKED",
                (stale_before, limit),
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

    def recover_queued_deliveries(self, *, limit: int = 100) -> tuple[Operation, ...]:
        """Bound recovery when delivery dies before the application can claim it.

        Scheduled retries count as live deliveries. A lost delivery consumes an
        application attempt before requeue, preventing unlimited repair loops.
        Delivery IDs remain when queue retention removes their corresponding jobs.
        """
        if not 1 <= limit <= 1000:
            raise ValueError("recovery limit must be 1..1000")
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT o.document FROM regimex_operations o WHERE o.state='queued' "
                "AND NOT EXISTS (SELECT 1 FROM regimex_deliveries d "
                "JOIN procrastinate_jobs j ON j.id=d.job_id "
                "WHERE d.operation_id=o.operation_id AND j.status IN ('todo','doing')) "
                "ORDER BY o.operation_id LIMIT %s FOR UPDATE OF o SKIP LOCKED",
                (limit,),
            ).fetchall()
            recovered: list[Operation] = []
            for row in rows:
                operation = _read_operation(row[0])
                token = uuid4()
                operation = start_attempt(operation, token, self._now(operation))
                operation = fail(
                    operation,
                    token,
                    OperationFailure(
                        code="infrastructure_failure",
                        message="The queue delivery ended before the operation could be claimed.",
                        retryable=True,
                    ),
                    self._now(operation),
                )
                self._save(connection, operation)
                if operation.state == "queued":
                    self._enqueue(connection, operation)
                recovered.append(operation)
            return tuple(recovered)
