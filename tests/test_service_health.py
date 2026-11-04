"""Real TLS readiness, queue heartbeats, worker death and private operational logs."""

import json
import logging
import os
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.request import Request, urlopen
from uuid import uuid4

import pytest

pytest.importorskip("psycopg")
pytest.importorskip("procrastinate")
pytest.importorskip("alembic")
pytest.importorskip("uvicorn")

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
from pydantic import SecretStr
from typer.testing import CliRunner

from regimex.cli import app
from regimex.infrastructure.database import DatabaseRepository
from regimex.service_health import api_healthy, queue_status, read_worker_identity, worker_healthy
from regimex.service_logging import RedactingJSONFormatter
from regimex.service_settings import ServiceSettings


@pytest.fixture
def database() -> Iterator[DatabaseRepository]:
    maintenance = os.environ.get("REGIMEX_TEST_DATABASE_DSN")
    if not maintenance:
        pytest.skip(
            "set a disposable PostgreSQL cluster DSN with CREATEDB and CREATEROLE "
            "for health process tests"
        )
    name = "regimex_health_" + uuid4().hex
    password = uuid4().hex
    with psycopg.connect(maintenance, autocommit=True) as connection:
        connection.execute(
            sql.SQL("CREATE ROLE {} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD {}").format(
                sql.Identifier(name), sql.Literal(password)
            )
        )
        try:
            connection.execute(
                sql.SQL("CREATE DATABASE {} OWNER {}").format(
                    sql.Identifier(name), sql.Identifier(name)
                )
            )
        except BaseException:
            connection.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(name)))
            raise
    repository = DatabaseRepository(
        make_conninfo(maintenance, dbname=name, user=name, password=password)
    )
    try:
        repository.migrate()
        with repository.connection() as connection:
            assert connection.execute(
                "SELECT rolsuper,rolcreatedb,rolcreaterole FROM pg_roles WHERE rolname=current_user"
            ).fetchone() == (False, False, False)
        yield repository
    finally:
        with psycopg.connect(maintenance, autocommit=True) as connection:
            connection.execute(
                sql.SQL("ALTER DATABASE {} ALLOW_CONNECTIONS true").format(sql.Identifier(name))
            )
            connection.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name))
            )
            connection.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(name)))


def settings(
    database: DatabaseRepository, root: Path, *, origin: str = "https://localhost"
) -> ServiceSettings:
    return ServiceSettings(
        database_dsn=SecretStr(database._dsn),
        object_root=root / "objects",
        work_root=root / "work",
        public_origin=origin,
        heartbeat_seconds=1,
        stale_seconds=3,
    )


def environment(value: ServiceSettings, root: Path) -> dict[str, str]:
    result = {key: entry for key, entry in os.environ.items() if not key.startswith("REGIMEX_")}
    dsn = root / "database-dsn"
    dsn.write_text(value.database_dsn.get_secret_value())
    dsn.chmod(0o600)
    result.update(
        REGIMEX_DATABASE_DSN_FILE=str(dsn),
        REGIMEX_OBJECT_ROOT=str(value.object_root),
        REGIMEX_WORK_ROOT=str(value.work_root),
        REGIMEX_PUBLIC_ORIGIN=value.public_origin,
        REGIMEX_HEARTBEAT_SECONDS="1",
        REGIMEX_STALE_SECONDS="3",
    )
    return result


@contextmanager
def running(root: Path, env: dict[str, str], *args: str) -> Iterator[subprocess.Popen[bytes]]:
    with (root / "service.log").open("ab") as log:
        child = subprocess.Popen(
            [sys.executable, "-m", "regimex", "service", *args],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
        )
        try:
            yield child
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
                    raise AssertionError("fixture process did not stop gracefully") from None


def healthy_until(value: ServiceSettings, child: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        assert child.poll() is None
        if worker_healthy(value, "analysis"):
            return
        time.sleep(0.05)
    raise AssertionError("worker did not publish a live registered heartbeat")


def test_exact_worker_heartbeat_stale_death_unregister_and_database_unavailable(
    database: DatabaseRepository, tmp_path: Path
) -> None:
    value = settings(database, tmp_path)
    env = environment(value, tmp_path)
    assert not worker_healthy(value, "analysis")
    with running(tmp_path, env, "worker", "--queue", "analysis") as child:
        healthy_until(value, child)
        identity = read_worker_identity(value, "analysis")
        with database.connection() as connection:
            assert connection.execute(
                "SELECT queue_name,slots FROM regimex_worker_queues WHERE worker_id=%s",
                (identity.worker_id,),
            ).fetchone() == ("analysis", 1)
        assert queue_status(database, "analysis", stale_seconds=3).state == "available"
        assert not worker_healthy(value, "admin")
        # Suspend this exact queue process. A different registered worker cannot
        # satisfy the probe; the actual queue-library heartbeat must become stale.
        child.send_signal(signal.SIGSTOP)
        try:
            deadline = time.monotonic() + 5
            while worker_healthy(value, "analysis") and time.monotonic() < deadline:
                time.sleep(0.05)
            assert not worker_healthy(value, "analysis")
            assert queue_status(database, "analysis", stale_seconds=3).state == "unavailable"
        finally:
            child.send_signal(signal.SIGCONT)
        healthy_until(value, child)
        maintenance = os.environ["REGIMEX_TEST_DATABASE_DSN"]
        database_name = psycopg.conninfo.conninfo_to_dict(database._dsn)["dbname"]
        assert isinstance(database_name, str)
        with psycopg.connect(maintenance, autocommit=True) as connection:
            connection.execute(
                sql.SQL("ALTER DATABASE {} ALLOW_CONNECTIONS false").format(
                    sql.Identifier(database_name)
                )
            )
            try:
                assert not worker_healthy(value, "analysis")
            finally:
                connection.execute(
                    sql.SQL("ALTER DATABASE {} ALLOW_CONNECTIONS true").format(
                        sql.Identifier(database_name)
                    )
                )
        healthy_until(value, child)
    assert child.returncode == 0 and not worker_healthy(value, "analysis")
    with database.connection() as connection:
        assert connection.execute("SELECT * FROM regimex_worker_queues").fetchall() == []
        assert (
            connection.execute(
                "SELECT 1 FROM procrastinate_workers WHERE id=%s", (identity.worker_id,)
            ).fetchone()
            is None
        )
    with running(tmp_path, env, "worker", "--queue", "analysis") as killed:
        healthy_until(value, killed)
        second = read_worker_identity(value, "analysis")
        assert second.worker_id != identity.worker_id
        killed.kill()
        killed.wait(timeout=5)
        # A fresh DB heartbeat left by an abruptly dead process is insufficient.
        assert not worker_healthy(value, "analysis")
        with database.connection() as connection:
            assert connection.execute(
                "SELECT 1 FROM procrastinate_workers WHERE id=%s", (second.worker_id,)
            ).fetchone() == (1,)
    log = (tmp_path / "service.log").read_text()
    assert value.database_dsn.get_secret_value() not in log


def test_queue_status_uses_actual_queue_capacity_and_heartbeat_freshness(
    database: DatabaseRepository,
) -> None:
    assert queue_status(database, "analysis", stale_seconds=3).state == "unavailable"
    with database.connection() as connection:
        unmanaged = connection.execute(
            "INSERT INTO procrastinate_workers DEFAULT VALUES RETURNING id"
        ).fetchone()
        assert unmanaged is not None
        connection.execute(
            "INSERT INTO regimex_worker_queues VALUES (%s,'admin',1)", (unmanaged[0],)
        )
    assert queue_status(database, "analysis", stale_seconds=3).state == "unavailable"
    with database.connection() as connection:
        worker = connection.execute(
            "INSERT INTO procrastinate_workers DEFAULT VALUES RETURNING id"
        ).fetchone()
        assert worker is not None
        connection.execute("INSERT INTO regimex_worker_queues VALUES (%s,'analysis',2)", worker)
        connection.execute(
            "INSERT INTO procrastinate_jobs (queue_name,task_name,status,worker_id) "
            "VALUES ('analysis','regimex.analysis','doing',%s)",
            worker,
        )
    assert queue_status(database, "analysis", stale_seconds=3).state == "available"
    with database.connection() as connection:
        connection.execute(
            "INSERT INTO procrastinate_jobs (queue_name,task_name,status,worker_id) "
            "VALUES ('analysis','regimex.analysis','doing',%s)",
            worker,
        )
    busy = queue_status(database, "analysis", stale_seconds=3)
    assert busy.state == "busy" and busy.worker_capability == "not_probed"
    assert str(worker[0]) not in busy.message
    with database.connection() as connection:
        free = connection.execute(
            "INSERT INTO procrastinate_workers DEFAULT VALUES RETURNING id"
        ).fetchone()
        assert free is not None
        connection.execute("INSERT INTO regimex_worker_queues VALUES (%s,'analysis',1)", free)
    assert queue_status(database, "analysis", stale_seconds=3).state == "available"
    with database.connection() as connection:
        connection.execute(
            "UPDATE procrastinate_workers SET last_heartbeat=now()-interval '5 seconds' "
            "WHERE id=%s",
            free,
        )
    assert queue_status(database, "analysis", stale_seconds=3).state == "busy"
    with database.connection() as connection:
        connection.execute(
            "UPDATE procrastinate_workers SET last_heartbeat=now()-interval '5 seconds' "
            "WHERE id=%s",
            worker,
        )
    assert queue_status(database, "analysis", stale_seconds=3).state == "unavailable"
    with database.connection() as connection:
        connection.execute("DELETE FROM procrastinate_workers")
        assert connection.execute("SELECT * FROM regimex_worker_queues").fetchall() == []
        connection.execute("DELETE FROM regimex_restore_state")
    assert queue_status(database, "analysis", stale_seconds=3).state == "unknown"


def test_queue_registry_migration_preserves_existing_resources_and_is_repeatable(
    database: DatabaseRepository,
) -> None:
    credential = database.create_token("fixture", "user")
    with database.connection() as connection:
        connection.execute("DROP TABLE regimex_worker_queues")
        connection.execute("UPDATE alembic_version SET version_num='0001_initial'")
    assert not database.readiness()
    assert queue_status(database, "analysis", stale_seconds=3).state == "unknown"
    database.migrate()
    database.migrate()
    assert database.readiness()
    assert database.authenticate(credential).owner_id == "fixture"


def test_service_status_authenticates_and_exposes_only_safe_queue_evidence(
    database: DatabaseRepository, tmp_path: Path
) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from regimex.api import create_app
    from regimex.service_bootstrap import create_service

    value = settings(database, tmp_path)
    credential = database.create_token("private-diagnostic-owner", "user")
    with TestClient(create_app(create_service(value))) as client:
        assert client.get("/api/v1/service-status").status_code == 401
        headers = {"Authorization": f"Bearer {credential}"}
        absent = client.get("/api/v1/service-status", headers=headers)
        assert absent.status_code == 200
        assert absent.json()["state"] == "unavailable"
        assert absent.headers["Cache-Control"] == "private, no-store"
        with database.connection() as connection:
            worker = connection.execute(
                "INSERT INTO procrastinate_workers DEFAULT VALUES RETURNING id"
            ).fetchone()
            assert worker is not None
            connection.execute("INSERT INTO regimex_worker_queues VALUES (%s,'analysis',1)", worker)
        available = client.get("/api/v1/service-status", headers=headers)
        assert available.status_code == 200
        assert available.json()["state"] == "available"
        assert available.json()["worker_capability"] == "not_probed"
        assert set(available.json()) == {"state", "checked_at", "message", "worker_capability"}
        for private in (credential, "private-diagnostic-owner", database._dsn, str(tmp_path)):
            assert private not in available.text


def test_doctor_reports_missing_model_and_workers_without_private_connection_details(
    database: DatabaseRepository, tmp_path: Path
) -> None:
    value = settings(database, tmp_path)
    env = environment(value, tmp_path)
    result = CliRunner().invoke(app, ["service", "doctor", "--json"], env=env)
    assert result.exit_code == 1, result.output
    report = json.loads(result.stdout)
    checks = {item["name"]: item for item in report["checks"]}
    assert report["healthy"] is False
    assert checks["database"]["status"] == checks["schema"]["status"] == "passed"
    assert checks["storage"]["status"] == "passed"
    assert checks["active_model"]["status"] == checks["analysis_queue"]["status"] == "failed"
    assert checks["model_load"]["status"] == "skipped"
    assert value.database_dsn.get_secret_value() not in result.output
    assert str(value.object_root) not in result.output


def certificate(root: Path, name: str) -> tuple[Path, Path]:
    executable = shutil.which("openssl")
    if executable is None:
        pytest.skip("openssl is required for actual TLS/SNI health tests")
    cert, key = root / (name + ".pem"), root / (name + ".key")
    subprocess.run(
        [
            executable,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost",
            "-addext",
            "basicConstraints=critical,CA:TRUE",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    key.chmod(0o600)
    return cert, key


def test_direct_tls_readiness_sni_ca_origin_and_redacted_request_logs(
    database: DatabaseRepository, tmp_path: Path
) -> None:
    with socket.socket() as selected:
        selected.bind(("127.0.0.1", 0))
        port = selected.getsockname()[1]
    value = settings(database, tmp_path, origin=f"https://localhost:{port}")
    env = environment(value, tmp_path)
    cert, key = certificate(tmp_path, "service")
    wrong, _ = certificate(tmp_path, "wrong")
    rejected = CliRunner().invoke(app, ["service", "serve"], env=env)
    assert rejected.exit_code == 1 and "direct TLS" in rejected.output
    with running(
        tmp_path,
        env,
        "serve",
        "--port",
        str(port),
        "--tls-cert-file",
        str(cert),
        "--tls-key-file",
        str(key),
    ) as child:
        deadline = time.monotonic() + 15
        while not api_healthy(value, component="ready", port=port, ca_file=cert):
            assert child.poll() is None and time.monotonic() < deadline
            time.sleep(0.05)
        assert not api_healthy(value, component="models", port=port, ca_file=cert)
        assert not api_healthy(value, component="ready", port=port, ca_file=wrong)
        assert not api_healthy(
            value.model_copy(update={"public_origin": f"https://wrong.example:{port}"}),
            component="ready",
            port=port,
            ca_file=cert,
        )
        request = Request(
            value.public_origin + "/health/live?token=private-query-marker",
            headers={"Authorization": "Bearer private-header-marker"},
        )
        with urlopen(
            request, context=ssl.create_default_context(cafile=str(cert)), timeout=5
        ) as response:
            assert response.status == 200
    log = (tmp_path / "service.log").read_text()
    assert "private-query-marker" not in log and "private-header-marker" not in log
    assert value.database_dsn.get_secret_value() not in log
    lines = [json.loads(line) for line in log.splitlines()]
    assert any(item.get("route") == "/health/live" and item.get("status") == 200 for item in lines)


def test_structured_log_redacts_credentials_without_hiding_scientific_warning() -> None:
    formatter = RedactingJSONFormatter(("private-dsn-value",))
    record = logging.LogRecord(
        "regimex.worker",
        logging.WARNING,
        __file__,
        1,
        (
            "ConvergenceWarning: optimizer reached its iteration limit; "
            "password=private-password; private-dsn-value"
        ),
        (),
        None,
    )
    record.payload = {"Authorization": "Bearer private-unlisted-extra"}
    result = formatter.format(record)
    assert "ConvergenceWarning: optimizer reached its iteration limit" in result
    assert not any(
        value in result
        for value in ("private-password", "private-dsn-value", "private-unlisted-extra")
    )


@pytest.mark.parametrize(
    "message",
    [
        "host=db.example user=fixture password='synthetic PASSWORD_TAIL_729'",
        'password="synthetic \\"quoted\\" PASSWORD_TAIL_729"',
        "Authorization: Bearer AUTH_SECRET_929",
        "Authorization: Basic AUTH_SECRET_929",
        '{"Authorization": "Bearer AUTH_SECRET_929"}',
        "AWS_SECRET_ACCESS_KEY='synthetic PASSWORD_TAIL_729'",
    ],
)
def test_structured_logs_redact_quoted_assignments_and_complete_authorization(message: str) -> None:
    formatter = RedactingJSONFormatter(
        ("postgresql://fixture:synthetic%20PASSWORD_TAIL_729@db.example/regimex",)
    )
    record = logging.LogRecord("worker", logging.ERROR, __file__, 1, message, (), None)
    record.action = message
    result = formatter.format(record)
    assert not any(secret in result for secret in ("PASSWORD_TAIL_729", "AUTH_SECRET_929"))
