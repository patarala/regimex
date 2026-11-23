"""Operational entry points remain discoverable in a core-only installation."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from regimex.cli import app


def test_service_help_keeps_optional_runtime_unloaded() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from typer.testing import CliRunner; from regimex.cli import app; "
            "r=CliRunner().invoke(app,['service','--help']); "
            "assert r.exit_code == 0, r.output; "
            "assert all(name in r.output for name in ('register-model','worker','doctor')); "
            "assert not {'fastapi','procrastinate','psycopg','numpy','sklearn','torch'} "
            "& sys.modules.keys()",
        ],
        check=True,
        env={**os.environ, "NO_COLOR": "1"},
    )


def test_service_needs_explicit_database_configuration() -> None:
    result = CliRunner().invoke(app, ["service", "migrate"], env={"REGIMEX_DATABASE_DSN": ""})
    assert result.exit_code == 1
    assert "REGIMEX_DATABASE_DSN" in result.output


@pytest.mark.parametrize("mode", ["defaults", "threads", "file", "conflict", "device"])
def test_register_model_inference_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    pytest.importorskip("psycopg")
    from regimex import service_bootstrap
    from regimex.infrastructure.admin_repository import AdminRepository
    from regimex.model_registry import InferenceRuntime

    token = tmp_path / "admin.token"
    token.write_text("fixture")
    service = Mock()
    monkeypatch.setattr(service_bootstrap, "create_service", lambda settings: service)
    monkeypatch.setattr(AdminRepository, "get_training_run_for_model", Mock(return_value=None))
    options = [
        "service",
        "register-model",
        "--bundle",
        str(tmp_path),
        "--manifest-sha256",
        "1" * 64,
        "--token-file",
        str(token),
    ]
    expected = InferenceRuntime()
    if mode in {"threads", "conflict"}:
        options.extend(["--threads", "3"])
        expected = InferenceRuntime(threads=3)
    if mode in {"file", "conflict", "device"}:
        config = tmp_path / "runtime.json"
        expected = InferenceRuntime(threads=1, max_batch_size=2, model_cache_budget_bytes=67108864)
        data = expected.model_dump()
        if mode == "device":
            data["device"] = "cuda"
        config.write_text(json.dumps(data))
        options.extend(["--runtime-file", str(config)])
    result = CliRunner().invoke(app, options, env={"REGIMEX_DATABASE_DSN": "dbname=not_connected"})
    if mode in {"conflict", "device"}:
        assert result.exit_code == 1
        assert ("threads" if mode == "conflict" else "device") in result.output
        service.register_bundle.assert_not_called()
    else:
        assert result.exit_code == 0, result.output
        registration = service.register_bundle.call_args.kwargs
        assert registration["inference_runtime"] == expected
        assert registration["record_training"] is True
        assert registration["activate"] is False
        assert registration["external_evaluations"] == ()


def test_register_model_imports_selected_final_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("psycopg")
    from regimex import service_bootstrap
    from regimex.infrastructure.admin_repository import AdminRepository
    from regimex.storage import digest, write_json

    token = tmp_path / "admin.token"
    token.write_text("fixture")
    sha = "1" * 64
    freeze = tmp_path / "freeze.json"
    native = tmp_path / "native.json"
    write_json(freeze, {"challenger": "forest", "bundle_sha256": sha, "protocol_sha256": "2" * 64})
    write_json(
        native,
        {
            "challenger": "forest",
            "freeze_sha256": digest(freeze),
            "experiments": 384,
            "native_qualified": True,
            "scores": {"forest": {"interval_macro_f1": 0.93, "point_macro_f1": 0.97}},
        },
    )
    service = Mock()
    monkeypatch.setattr(service_bootstrap, "create_service", lambda settings: service)
    find_run = Mock(return_value=Mock(run_id="f6aea2bc-ef47-46c9-b39a-1a80ba058e31"))
    monkeypatch.setattr(AdminRepository, "get_training_run_for_model", find_run)
    args = [
        "service",
        "register-model",
        "--bundle",
        str(tmp_path),
        "--token-file",
        str(token),
        "--manifest-sha256",
        sha,
        "--selection-file",
        str(freeze),
        "--native-final",
        str(native),
        "--name",
        "Offline forest",
    ]
    env = {"REGIMEX_DATABASE_DSN": "dbname=not_connected"}
    result = CliRunner().invoke(app, args, env=env)
    assert result.exit_code == 0, result.output
    assert "/training/f6aea2bc-ef47-46c9-b39a-1a80ba058e31" in result.stdout
    find_run.assert_called_once_with(
        service.authenticate.return_value, service.register_bundle.return_value.model_version_id
    )
    registration = service.register_bundle.call_args.kwargs
    assert registration["display_name"] == "Offline forest"
    (report,) = registration["external_evaluations"]
    assert report.kind == "native_final" and report.sample_count == 384
    assert report.report_sha256 == digest(native) and report.interval_macro_f1 == 0.93
    write_json(freeze, {"challenger": "other", "bundle_sha256": sha, "protocol_sha256": "2" * 64})
    result = CliRunner().invoke(app, args, env=env)
    assert result.exit_code == 1 and "different selection" in result.output
    service.register_bundle.assert_called_once()


def test_database_failure_does_not_print_connection_credentials() -> None:
    pytest.importorskip("psycopg")
    pytest.importorskip("sqlalchemy")
    result = CliRunner().invoke(
        app,
        ["service", "migrate"],
        env={"REGIMEX_DATABASE_DSN": "postgresql://user:private-example@127.0.0.1:1/missing"},
    )
    assert result.exit_code == 1
    assert "infrastructure" in result.output
    assert "private-example" not in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("structured", [False, True])
def test_doctor_reports_database_failure_without_credentials(
    tmp_path: Path, structured: bool
) -> None:
    pytest.importorskip("psycopg")
    pytest.importorskip("sqlalchemy")
    result = CliRunner().invoke(
        app,
        ["service", "doctor", *(["--json"] if structured else [])],
        env={
            "REGIMEX_DATABASE_DSN": "postgresql://user:private-doctor-value@127.0.0.1:1/missing",
            "REGIMEX_OBJECT_ROOT": str(tmp_path / "objects"),
            "REGIMEX_WORK_ROOT": str(tmp_path / "work"),
        },
    )
    assert result.exit_code == 1
    if structured:
        report = json.loads(result.stdout)
        checks = {item["name"]: item["status"] for item in report["checks"]}
        assert checks["database"] == "failed"
        assert checks["schema"] == checks["model_load"] == "skipped"
        assert checks["storage"] == "passed"
    else:
        assert "FAIL database: PostgreSQL is unavailable." in result.stdout
        assert "PASS storage:" in result.stdout
        assert "SKIP model_load:" in result.stdout
    assert "private-doctor-value" not in result.output
    assert "Traceback" not in result.output
    assert "Error: 1" not in result.output


def test_http_development_server_rejects_a_public_listener() -> None:
    pytest.importorskip("uvicorn")
    pytest.importorskip("psycopg")
    result = CliRunner().invoke(
        app,
        ["service", "serve", "--host", "0.0.0.0"],
        env={
            "REGIMEX_DATABASE_DSN": "dbname=not_connected",
            "REGIMEX_PUBLIC_ORIGIN": "http://127.0.0.1:8000",
            "REGIMEX_ALLOW_LOOPBACK_HTTP": "true",
        },
    )
    assert result.exit_code == 1 and "literal loopback listener" in result.output


def test_https_server_rejects_a_readable_private_key_before_startup(tmp_path: Path) -> None:
    pytest.importorskip("uvicorn")
    pytest.importorskip("psycopg")
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert.write_text("deliberate validation fixture")
    key.write_text("deliberate private key placeholder")
    key.chmod(0o644)
    result = CliRunner().invoke(
        app,
        ["service", "serve", "--tls-cert-file", str(cert), "--tls-key-file", str(key)],
        env={
            "REGIMEX_DATABASE_DSN": "dbname=not_connected",
            "REGIMEX_PUBLIC_ORIGIN": "https://localhost",
            "REGIMEX_ALLOW_LOOPBACK_HTTP": "false",
        },
    )
    assert result.exit_code == 1 and "private file" in result.output


@pytest.mark.parametrize(
    ("active_limit", "connection_limit", "expected"),
    [(None, None, 32), ("16", None, 40), ("6", "18", 18)],
)
def test_server_installs_intake_deadlines_without_automatic_protocol_selection(
    monkeypatch: pytest.MonkeyPatch,
    active_limit: str | None,
    connection_limit: str | None,
    expected: int,
) -> None:
    uvicorn = pytest.importorskip("uvicorn")
    pytest.importorskip("psycopg")
    from regimex import service_bootstrap

    application = object()
    captured: dict[str, Any] = {}

    def run(app: object, **options: Any) -> None:
        assert app is application
        captured.update(options)

    monkeypatch.setattr(service_bootstrap, "create_app", lambda: application)
    monkeypatch.setattr(uvicorn, "run", run)
    for variable, value in (
        ("REGIMEX_HTTP_CONCURRENCY_LIMIT", active_limit),
        ("REGIMEX_HTTP_CONNECTION_LIMIT", connection_limit),
    ):
        if value is None:
            monkeypatch.delenv(variable, raising=False)
        else:
            monkeypatch.setenv(variable, value)
    result = CliRunner().invoke(
        app,
        ["service", "serve"],
        env={
            "REGIMEX_DATABASE_DSN": "dbname=not_connected",
            "REGIMEX_PUBLIC_ORIGIN": "http://127.0.0.1:8000",
            "REGIMEX_ALLOW_LOOPBACK_HTTP": "true",
        },
    )
    assert result.exit_code == 0, result.output
    assert isinstance(captured["http"], type)
    assert captured["ws"] == "none"
    assert captured["limit_concurrency"] == expected
    assert captured["proxy_headers"] is False


def test_local_storage_check_reads_inventory_without_contacting_the_database(
    tmp_path: Path,
) -> None:
    pytest.importorskip("psycopg")
    pytest.importorskip("sqlalchemy")
    from regimex.infrastructure.object_store import LocalObjectStore

    source = tmp_path / "payload.bin"
    source.write_bytes(b"abc")
    root = tmp_path / "objects"
    LocalObjectStore(root, max_object_bytes=512 * 1024**2, max_total_bytes=4 * 1024**3).put(
        "deliberate-fixture/payload.bin", source, expected_sha256=hashlib.sha256(b"abc").hexdigest()
    )
    result = CliRunner().invoke(
        app,
        ["service", "storage-check"],
        env={
            "REGIMEX_DATABASE_DSN": "dbname=not_connected",
            "REGIMEX_OBJECT_ROOT": str(root),
            "REGIMEX_WORK_ROOT": str(tmp_path / "work"),
        },
    )
    assert result.exit_code == 0, result.output
    inventory = json.loads(result.output)
    assert inventory["backend"] == "local"
    assert inventory["retained_versions"] == 1 and inventory["retained_payload_bytes"] == 3
    assert inventory["scope"] == "configuration_and_inventory"
    assert "deliberate-fixture" not in result.output


@pytest.mark.parametrize("ready", [True, False])
def test_database_health_checks_the_packaged_schema(
    monkeypatch: pytest.MonkeyPatch, ready: bool
) -> None:
    pytest.importorskip("psycopg")
    from regimex.infrastructure.database import DatabaseRepository

    probe = Mock(return_value=ready)
    monkeypatch.setattr(DatabaseRepository, "readiness", probe)
    result = CliRunner().invoke(
        app,
        ["service", "health", "--component", "database"],
        env={"REGIMEX_DATABASE_DSN": "dbname=offline_test"},
    )
    assert result.exit_code == (0 if ready else 1), result.output
    assert json.loads(result.stdout)["status"] == ("healthy" if ready else "unhealthy")
    probe.assert_called_once_with()
