"""Offline coordinated backups and restoration into a new empty deployment.

Stopped service/worker/maintenance processes and exclusive store access remain
operator requirements. SQL locks and immutable objects provide extra checks;
this module makes no online backup or distributed transaction claim.
"""

import hashlib
import os
import re
import signal
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.types.json import Json, Jsonb
from pydantic import JsonValue, SecretStr, TypeAdapter, ValidationError

from regimex.admin_contracts import DatasetRecord, EvaluationRecord, ReferenceSetRecord
from regimex.application.service import materialize_object
from regimex.contracts import StrictModel
from regimex.infrastructure.maintenance import (
    Connection,
    ServicesStoppedAttestation,
    maintenance_lock,
    quiescent_transaction,
)
from regimex.infrastructure.object_store import ObjectInfo, ObjectStore
from regimex.recovery_contracts import (
    RECOVERY_REVISION,
    BackupManifest,
    BackupResult,
    MaintenanceIdentity,
    PostgresTools,
    RecoveryLimits,
    RestoreReceipt,
    TableEvidence,
)
from regimex.report_contracts import AnalysisReportRecord
from regimex.service_contracts import ModelRecord, SnapshotRecord
from regimex.storage import json_bytes

_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]{0,62}\Z")
_VERSION_FIELDS: dict[str, tuple[str, type[StrictModel]]] = {
    "regimex_snapshots": ("snapshot_id", SnapshotRecord),
    "regimex_models": ("model_version_id", ModelRecord),
    "regimex_datasets": ("dataset_id", DatasetRecord),
    "regimex_reference_sets": ("reference_set_id", ReferenceSetRecord),
    "regimex_evaluation_records": ("run_id", EvaluationRecord),
    "regimex_analysis_reports": ("analysis_id", AnalysisReportRecord),
}
_DEFAULT_LIMITS = RecoveryLimits()


def _safe_recovery[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    @wraps(function)
    def checked(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return function(*args, **kwargs)
        except ValidationError:
            raise ValueError("Recovery metadata failed schema validation.") from None
        except psycopg.Error:
            raise ValueError(
                "PostgreSQL recovery failed; the target must remain offline."
            ) from None

    return checked


def _json_object(value: object) -> dict[str, JsonValue]:
    return _JSON_OBJECT.validate_python(value)


def _object(value: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError("Recovery metadata requires an object.")
    return value


def _list(value: JsonValue) -> list[JsonValue]:
    if not isinstance(value, list):
        raise ValueError("Recovery metadata requires a list.")
    return value


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Recovery metadata requires a string.")
    return value


def _first(connection: Connection, query: str) -> object:
    row = connection.execute(query).fetchone()
    if row is None:
        raise ValueError("Required recovery metadata is absent.")
    return row[0]


def _deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise TimeoutError("Offline recovery exceeded its wall-time budget.")


def _attestation(value: ServicesStoppedAttestation) -> ServicesStoppedAttestation:
    result = ServicesStoppedAttestation.model_validate_json(value.model_dump_json())
    if result.stopped_at > datetime.now(UTC):
        raise ValueError("The stopped-services attestation cannot come from the future.")
    return result


def _tool_version(executable: Path) -> str:
    return subprocess.run(
        [str(executable), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


def _postgres_major(value: str) -> str:
    version = re.search(r"\b([1-9][0-9]*)\.", value)
    if version is None:
        raise ValueError("A PostgreSQL tool did not report a supported version.")
    return version.group(1)


def _stream_digest(stream: BinaryIO, deadline: float) -> str:
    checksum_value = hashlib.sha256()
    while block := stream.read(1024 * 1024):
        _deadline(deadline)
        checksum_value.update(block)
    _deadline(deadline)
    return checksum_value.hexdigest()


def checksum(path: Path, *, deadline: float) -> str:
    with path.open("rb") as stream:
        return _stream_digest(stream, deadline)


def verified_file(path: Path, *, maximum: int, size: int, sha256: str, deadline: float) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != size or size > maximum:
            raise ValueError("A backup member is not a bounded regular file of the recorded size.")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            actual = _stream_digest(stream, deadline)
        after = os.fstat(descriptor)
        if actual != sha256 or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("A backup member failed its immutable byte check.")
    finally:
        os.close(descriptor)


def _private_write(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read(path: Path, maximum: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Backup members must be regular files.")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(maximum + 1)
        if len(data) > maximum:
            raise ValueError("A backup member exceeds its byte limit.")
        return data
    finally:
        os.close(descriptor)


def _libpq_environment(dsn: SecretStr) -> dict[str, str]:
    # Explicitly clear inherited libpq defaults. Passwords remain out of argv,
    # manifests and persisted logs. The maintenance process still holds secrets.
    environment = {key: value for key, value in os.environ.items() if not key.startswith("PG")}
    names = {
        "host": "PGHOST",
        "hostaddr": "PGHOSTADDR",
        "port": "PGPORT",
        "dbname": "PGDATABASE",
        "user": "PGUSER",
        "password": "PGPASSWORD",
        "passfile": "PGPASSFILE",
        "service": "PGSERVICE",
        "sslmode": "PGSSLMODE",
        "sslcert": "PGSSLCERT",
        "sslkey": "PGSSLKEY",
        "sslrootcert": "PGSSLROOTCERT",
        "sslcrl": "PGSSLCRL",
        "sslcrldir": "PGSSLCRLDIR",
        "sslpassword": "PGSSLPASSWORD",
        "channel_binding": "PGCHANNELBINDING",
        "gssencmode": "PGGSSENCMODE",
        "target_session_attrs": "PGTARGETSESSIONATTRS",
        "options": "PGOPTIONS",
        "connect_timeout": "PGCONNECT_TIMEOUT",
        "application_name": "PGAPPNAME",
    }
    for key, value in conninfo_to_dict(dsn.get_secret_value()).items():
        if key not in names:
            raise ValueError("This libpq setting is not supported by offline recovery.")
        if value is not None:
            environment[names[key]] = str(value)
    environment["PGAPPNAME"] = "regimex-offline-recovery"
    environment["PGCONNECT_TIMEOUT"] = "5"
    return environment


def _run_tool(
    command: list[str],
    dsn: SecretStr,
    *,
    deadline: float,
    output: Path | None = None,
    stdout: BinaryIO | None = None,
    output_limit: int,
) -> None:
    # A private temporary stderr is deleted; raised errors never include libpq's
    # raw connection diagnostics or the credential-bearing environment.
    with tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(
            command,
            env=_libpq_environment(dsn),
            stdout=subprocess.DEVNULL if stdout is None else stdout,
            stderr=stderr,
            start_new_session=True,
        )

        def within_output_limit() -> None:
            size = output.stat().st_size if output is not None else 0
            if stdout is not None:
                size += os.fstat(stdout.fileno()).st_size
            if size > output_limit:
                raise ValueError("A PostgreSQL recovery output exceeds its byte limit.")

        try:
            while process.poll() is None:
                _deadline(deadline)
                within_output_limit()
                time.sleep(0.05)
            if process.returncode:
                raise RuntimeError("The PostgreSQL recovery tool failed; no target is ready.")
            within_output_limit()
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)


def _schema(value: str) -> str:
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError("Recovery requires a bounded lowercase PostgreSQL schema identifier.")
    return value


def _tables(connection: Connection, schema: str) -> tuple[str, ...]:
    return tuple(
        _text(row[0])
        for row in connection.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname=%s ORDER BY tablename", (schema,)
        )
    )


def _set_path(connection: Connection, schema: str) -> None:
    connection.execute(
        sql.SQL("SET LOCAL search_path TO {}").format(sql.Identifier(_schema(schema)))
    )


def _rewrite_document(
    table: str,
    document: dict[str, JsonValue],
    replacements: dict[str, ObjectInfo],
    old: dict[str, ObjectInfo],
) -> dict[str, JsonValue]:
    model = _VERSION_FIELDS[table][1]
    result = _json_object(model.model_validate_json(json_bytes(document)).model_dump(mode="json"))
    locations: list[tuple[dict[str, JsonValue], str]] = []
    if table in {"regimex_snapshots", "regimex_analysis_reports"}:
        locations = [(result, "object")]
    elif table == "regimex_models":
        members = _object(result["members"])
        locations = [(result, "manifest"), *((members, key) for key in members)]
    elif table == "regimex_datasets":
        if result["generation_provenance"] is not None:
            locations = [(result, "generation_provenance")]
    elif table == "regimex_reference_sets":
        locations = [(_object(item), "object") for item in _list(result["annotations"])]
    elif table == "regimex_evaluation_records":
        locations = [(result, "report")]
    for container, key in locations:
        info = ObjectInfo.model_validate(container[key])
        if info.key not in old or old[info.key] != info:
            raise ValueError("An embedded object reference differs from the backup object ledger.")
        container[key] = replacements[info.key].model_dump(mode="json")
    return _json_object(model.model_validate_json(json_bytes(result)).model_dump(mode="json"))


def _table_evidence(
    connection: Connection,
    schema: str,
    old: dict[str, ObjectInfo],
    limits: RecoveryLimits,
    *,
    deadline: float,
    added_claims: frozenset[tuple[str, str]] = frozenset(),
    pending_restore: bool = False,
) -> dict[str, TableEvidence]:
    # Sort fixed-size per-row digests, rather than retaining full analysis JSON
    # rows. Only explicitly whitelisted location/version fields are normalized.
    portable = {
        key: value.model_copy(update={"version_id": "portable-content"})
        for key, value in old.items()
    }
    evidence = {}
    revision = _first(connection, "SELECT version_num FROM alembic_version")
    if revision != RECOVERY_REVISION:
        raise ValueError("Offline recovery requires the current database schema revision.")
    for table in _tables(connection, schema):
        if table == "regimex_restore_state" and pending_restore:
            if connection.execute("SELECT 1 FROM regimex_restore_state LIMIT 1").fetchone():
                raise ValueError("The restored database guard became ready before verification.")
            row_digest = hashlib.sha256(json_bytes({"singleton": True, "ready": True})).digest()
            evidence[table] = TableEvidence(
                rows=1, portable_sha256=hashlib.sha256(row_digest).hexdigest()
            )
            continue
        hashes: list[bytes] = []
        with connection.cursor(name="backup_rows_" + uuid4().hex) as cursor:
            # A saved analysis can be 16 MiB. The default 100-row fetch would
            # otherwise defeat bounded backup memory on a populated service.
            cursor.itersize = 1
            cursor.execute(
                sql.SQL("SELECT row_to_json(t) FROM {}.{} t").format(
                    sql.Identifier(schema), sql.Identifier(table)
                )
            )
            for (raw,) in cursor:
                _deadline(deadline)
                row = _json_object(raw)
                if (
                    table == "regimex_object_version_claims"
                    and (_text(row["object_key"]), _text(row["version_id"])) in added_claims
                ):
                    if row["state"] != "registered":
                        raise ValueError("Restoration created an unexpected version claim.")
                    continue
                if len(hashes) >= limits.max_rows_per_table:
                    raise ValueError("A database table exceeds its recovery row limit.")
                if table == "regimex_objects":
                    info = ObjectInfo.model_validate(row["document"])
                    if old.get(info.key) != info or (
                        row["object_key"],
                        row["sha256"],
                        row["size_bytes"],
                        row["version_id"],
                    ) != (info.key, info.sha256, info.size_bytes, info.version_id):
                        raise ValueError("Object ledger columns/document differ from the manifest.")
                    row["version_id"] = "portable-content"
                    row["document"] = portable[info.key].model_dump(mode="json")
                elif table in _VERSION_FIELDS:
                    row["document"] = _rewrite_document(
                        table, _object(row["document"]), portable, old
                    )
                elif table == "regimex_maintenance_identity":
                    # The source value is separately pinned in the manifest. Only
                    # this physical deployment singleton rotates during restore;
                    # old review plans, tombstones and events remain unchanged.
                    row = {"singleton": row["singleton"], "physical_identity": "portable"}
                hashes.append(hashlib.sha256(json_bytes(row)).digest())
        checksum_value = hashlib.sha256(b"".join(sorted(hashes))).hexdigest()
        evidence[table] = TableEvidence(rows=len(hashes), portable_sha256=checksum_value)
    return evidence


@_safe_recovery
def backup_service(
    database_dsn: SecretStr,
    schema: str,
    store: ObjectStore,
    destination: Path,
    *,
    tools: PostgresTools,
    attestation: ServicesStoppedAttestation,
    limits: RecoveryLimits = _DEFAULT_LIMITS,
) -> BackupResult:
    schema = _schema(schema)
    deadline = time.monotonic() + limits.timeout_seconds
    attestation = _attestation(attestation)
    if destination.exists() or destination.is_symlink():
        raise ValueError("A backup destination must be new; existing paths are never overwritten.")
    destination.mkdir(mode=0o700)
    (destination / "objects").mkdir(mode=0o700)
    dump = destination / "database.dump"
    # pg_dump opens the file itself; a private parent plus umask protects bytes.
    descriptor = os.open(dump, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    with psycopg.connect(
        database_dsn.get_secret_value(), connect_timeout=5, autocommit=True
    ) as connection:
        maintenance_lock(connection, schema)
        with quiescent_transaction(connection, schema=schema, maximum_idle_seconds=0):
            revision = _text(_first(connection, "SELECT version_num FROM alembic_version"))
            if revision != RECOVERY_REVISION:
                raise ValueError("Offline recovery requires the current database schema revision.")
            rows = connection.execute(
                "SELECT document FROM regimex_objects ORDER BY object_key LIMIT %s",
                (limits.max_objects + 1,),
            ).fetchall()
            if len(rows) > limits.max_objects:
                raise ValueError("The backup exceeds its object-count limit.")
            objects = tuple(ObjectInfo.model_validate(row[0]) for row in rows)
            if sum(info.size_bytes for info in objects) > limits.max_total_object_bytes:
                raise ValueError("Registered objects exceed the backup byte limit.")
            old = {info.key: info for info in objects}
            table_evidence = _table_evidence(connection, schema, old, limits, deadline=deadline)
            if connection.execute(
                "SELECT singleton,ready FROM regimex_restore_state"
            ).fetchall() != [(True, True)]:
                raise ValueError("The source database has no completed restore guard.")
            source_identity = MaintenanceIdentity.model_validate_json(
                json_bytes(
                    _first(
                        connection,
                        "SELECT to_jsonb(t) FROM regimex_maintenance_identity t WHERE singleton",
                    )
                )
            )
            snapshot = _text(_first(connection, "SELECT pg_export_snapshot()"))
            version = _text(_first(connection, "SHOW server_version"))
            tool_version = _tool_version(tools.pg_dump)
            if _postgres_major(version) != _postgres_major(tool_version):
                raise ValueError(
                    "Offline backup requires a pg_dump matching the server major version."
                )
            _run_tool(
                [
                    str(tools.pg_dump),
                    "--format=custom",
                    "--no-owner",
                    "--no-privileges",
                    "--schema=" + schema,
                    "--snapshot=" + snapshot,
                    "--file=" + str(dump),
                ],
                database_dsn,
                deadline=deadline,
                output_limit=limits.max_dump_bytes,
                output=dump,
            )
            for info in objects:
                _deadline(deadline)
                path = destination / "objects" / info.sha256
                # Different logical keys can share a content digest. Verify every
                # source revision, even when one archive payload already exists.
                with tempfile.TemporaryDirectory(dir=destination, prefix=".copy-") as temporary:
                    copy = Path(temporary) / "payload"
                    materialize_object(store, info, copy)
                    copy.chmod(0o600)
                    with copy.open("rb") as stream:
                        os.fsync(stream.fileno())
                    if not path.exists():
                        os.link(copy, path)
                _deadline(deadline)
    manifest = BackupManifest(
        backup_id=uuid4(),
        schema_name=schema,
        migration_revision=RECOVERY_REVISION,
        created_at=datetime.now(UTC),
        attestation=attestation,
        postgres_version=version,
        dump_tool_version=tool_version,
        dump_size_bytes=dump.stat().st_size,
        dump_sha256=checksum(dump, deadline=deadline),
        objects=objects,
        tables=table_evidence,
        source_maintenance_identity=source_identity,
    )
    data = json_bytes(manifest.model_dump(mode="json"))
    if len(data) > limits.max_manifest_bytes:
        raise ValueError("The backup manifest exceeds its byte limit.")
    _private_write(destination / "manifest.json", data)
    with dump.open("rb") as stream:
        os.fsync(stream.fileno())
    for directory in (destination / "objects", destination, destination.parent):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return BackupResult(manifest=manifest, manifest_sha256=hashlib.sha256(data).hexdigest())


def _empty_database(connection: Connection) -> None:
    if connection.execute(
        "SELECT 1 FROM pg_namespace "
        "WHERE nspname NOT IN ('public','pg_catalog','information_schema') "
        "AND nspname NOT LIKE 'pg_toast%' AND nspname NOT LIKE 'pg_temp_%' LIMIT 1"
    ).fetchone():
        raise ValueError("Restore requires a newly provisioned empty database.")
    if connection.execute(
        "SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
        "WHERE n.nspname='public' LIMIT 1"
    ).fetchone():
        raise ValueError("Restore refuses existing database objects.")
    if connection.execute(
        "SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
        "WHERE n.nspname='public' LIMIT 1"
    ).fetchone():
        raise ValueError("Restore refuses existing database routines.")
    if (
        connection.execute(
            "SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid=t.typnamespace "
            "WHERE n.nspname='public' LIMIT 1"
        ).fetchone()
        or connection.execute("SELECT 1 FROM pg_largeobject_metadata LIMIT 1").fetchone()
    ):
        raise ValueError("Restore refuses existing user types or PostgreSQL large objects.")


def _registered_claims(
    connection: Connection, replacements: dict[str, ObjectInfo]
) -> frozenset[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    values = list(replacements.values())
    for start in range(0, len(values), 1000):
        batch = values[start : start + 1000]
        for key, version, state in connection.execute(
            "SELECT c.object_key,c.version_id,c.state FROM regimex_object_version_claims c "
            "JOIN unnest(%s::text[],%s::text[]) AS n(object_key,version_id) "
            "ON (c.object_key,c.version_id)=(n.object_key,n.version_id)",
            ([item.key for item in batch], [item.version_id for item in batch]),
        ):
            if state != "registered":
                raise ValueError(
                    "A restored live object generation collides with a deletion fence."
                )
            found.add((_text(key), _text(version)))
    return frozenset(found)


@_safe_recovery
def restore_service(
    target_dsn: SecretStr,
    store: ObjectStore,
    backup: Path,
    *,
    expected_manifest_sha256: str,
    assert_empty_store: Callable[[], None],
    tools: PostgresTools,
    attestation: ServicesStoppedAttestation,
    limits: RecoveryLimits = _DEFAULT_LIMITS,
) -> RestoreReceipt:
    attestation = _attestation(attestation)
    deadline = time.monotonic() + limits.timeout_seconds
    data = _read(backup / "manifest.json", limits.max_manifest_bytes)
    if hashlib.sha256(data).hexdigest() != expected_manifest_sha256:
        raise ValueError("The backup manifest differs from the trusted external digest.")
    manifest = BackupManifest.model_validate_json(data)
    schema = _schema(manifest.schema_name)
    if (
        len(manifest.objects) > limits.max_objects
        or sum(item.size_bytes for item in manifest.objects) > limits.max_total_object_bytes
    ):
        raise ValueError("Backup object inventory exceeds restore limits.")
    dump = backup / "database.dump"
    verified_file(
        dump,
        maximum=limits.max_dump_bytes,
        size=manifest.dump_size_bytes,
        sha256=manifest.dump_sha256,
        deadline=deadline,
    )
    for info in manifest.objects:
        path = backup / "objects" / info.sha256
        verified_file(
            path,
            maximum=limits.max_total_object_bytes,
            size=info.size_bytes,
            sha256=info.sha256,
            deadline=deadline,
        )
        if time.monotonic() > deadline:
            raise TimeoutError("Restore exceeded its overall wall-time budget.")
    with psycopg.connect(
        target_dsn.get_secret_value(), connect_timeout=5, autocommit=True
    ) as guard:
        maintenance_lock(guard, schema)
        _empty_database(guard)
        if _postgres_major(_text(_first(guard, "SHOW server_version"))) != _postgres_major(
            manifest.postgres_version
        ) or _postgres_major(_tool_version(tools.pg_restore)) != _postgres_major(
            manifest.dump_tool_version
        ):
            raise ValueError("Offline restore requires matching PostgreSQL server and tool majors.")
        assert_empty_store()
        replacements = {}
        for info in manifest.objects:
            if time.monotonic() > deadline:
                raise TimeoutError("Restore exceeded its overall wall-time budget.")
            restored = store.put(
                info.key, backup / "objects" / info.sha256, expected_sha256=info.sha256
            )
            if (restored.key, restored.sha256, restored.size_bytes) != (
                info.key,
                info.sha256,
                info.size_bytes,
            ):
                raise ValueError("The target object adapter changed object content identity.")
            replacements[info.key] = restored
            with tempfile.TemporaryDirectory(prefix="regimex-restored-object-") as temporary:
                materialize_object(store, restored, Path(temporary) / "verified")
            _deadline(deadline)
        target_database = make_conninfo(
            dbname=conninfo_to_dict(target_dsn.get_secret_value())["dbname"]
        )
        command = [
            str(tools.pg_restore),
            "--no-owner",
            "--no-privileges",
            "--single-transaction",
            "--exit-on-error",
            "--dbname=" + target_database,
        ]
        # A pristine database already contains public. pg_restore has no schema
        # remapping option; preserve the original schema and skip only its CREATE
        # SCHEMA TOC entry, never rewrite SQL text or archived data.
        with tempfile.TemporaryDirectory(prefix="regimex-restore-toc-") as temporary:
            with tempfile.TemporaryFile() as raw_toc:
                _run_tool(
                    [str(tools.pg_restore), "--list", str(dump)],
                    target_dsn,
                    stdout=raw_toc,
                    deadline=deadline,
                    output_limit=limits.max_manifest_bytes,
                )
                raw_toc.seek(0)
                lines = raw_toc.read().decode("utf-8").splitlines(keepends=True)
            kept = []
            excluded_guard = 0
            for line in lines:
                if schema == "public" and re.match(
                    r"^[0-9]+; [0-9]+ [0-9]+ SCHEMA - public ", line
                ):
                    continue
                if re.match(
                    r"^[0-9]+; [0-9]+ [0-9]+ TABLE DATA " + schema + r" regimex_restore_state ",
                    line,
                ):
                    excluded_guard += 1
                    continue
                kept.append(line)
            if excluded_guard != 1:
                raise ValueError(
                    "The archive does not contain exactly one restore-guard data entry."
                )
            toc = Path(temporary) / "restore.list"
            _private_write(toc, "".join(kept).encode())
            command.append("--use-list=" + str(toc))
            _run_tool(
                [*command, str(dump)],
                target_dsn,
                deadline=deadline,
                output_limit=limits.max_dump_bytes,
            )
        old = {info.key: info for info in manifest.objects}
        with quiescent_transaction(guard, schema=schema, maximum_idle_seconds=0):
            connection = guard
            if (
                _table_evidence(
                    connection, schema, old, limits, pending_restore=True, deadline=deadline
                )
                != manifest.tables
            ):
                raise ValueError(
                    "The restored database archive differs from its pinned logical rows."
                )
            current_identity = MaintenanceIdentity.model_validate_json(
                json_bytes(
                    _first(
                        connection,
                        "SELECT to_jsonb(t) FROM regimex_maintenance_identity t WHERE singleton",
                    )
                )
            )
            if current_identity != manifest.source_maintenance_identity:
                raise ValueError("The copied maintenance identity differs from the backup.")
            new_installation = uuid4()
            connection.execute(
                "UPDATE regimex_maintenance_identity SET installation_id=%s,created_at=now(),"
                "restored_from_installation_id=%s,restored_from_backup_sha256=%s "
                "WHERE singleton",
                (
                    new_installation,
                    current_identity.installation_id,
                    expected_manifest_sha256,
                ),
            )
            all_claims = frozenset((info.key, info.version_id) for info in replacements.values())
            added_claims = all_claims - _registered_claims(connection, replacements)
            for key, info in replacements.items():
                _deadline(deadline)
                connection.execute(
                    "UPDATE regimex_objects SET version_id=%s,document=%s WHERE object_key=%s",
                    (info.version_id, Jsonb(info.model_dump(mode="json")), key),
                )
            for table, (primary, _) in _VERSION_FIELDS.items():
                with connection.cursor(name="restore_records_" + uuid4().hex) as rows:
                    rows.itersize = 1
                    rows.execute(
                        sql.SQL("SELECT {},document FROM {}").format(
                            sql.Identifier(primary), sql.Identifier(table)
                        )
                    )
                    for identifier, document in rows:
                        _deadline(deadline)
                        replacement = _rewrite_document(
                            table, _json_object(document), replacements, old
                        )
                        connection.execute(
                            sql.SQL("UPDATE {} SET document=%s WHERE {}=%s").format(
                                sql.Identifier(table), sql.Identifier(primary)
                            ),
                            (Json(replacement), identifier),
                        )
            if _registered_claims(connection, replacements) != all_claims:
                raise ValueError("Restored live objects lack their registered generation claims.")
            current = _table_evidence(
                connection,
                schema,
                replacements,
                limits,
                added_claims=added_claims,
                pending_restore=True,
                deadline=deadline,
            )
            if current != manifest.tables:
                raise ValueError(
                    "Restored logical rows differ; keep the target offline and discard it."
                )
            receipt = RestoreReceipt(
                backup_id=manifest.backup_id,
                backup_manifest_sha256=expected_manifest_sha256,
                restored_at=datetime.now(UTC),
                attestation=attestation,
                schema_name=schema,
                object_count=len(replacements),
                changed_versions=sum(
                    old[key].version_id != value.version_id for key, value in replacements.items()
                ),
                table_evidence=current,
                new_installation_id=new_installation,
                new_registered_claims=len(added_claims),
                new_claims_sha256=hashlib.sha256(json_bytes(sorted(added_claims))).hexdigest()
                if added_claims
                else None,
            )
            connection.execute(
                "INSERT INTO regimex_restore_state(singleton,ready) VALUES (true,true)"
            )
    return receipt
