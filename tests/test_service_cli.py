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
    from regimex.model_registry import InferenceRuntime

    token = tmp_path / "admin.token"
    token.write_text("fixture")
    service = Mock()
    monkeypatch.setattr(service_bootstrap, "create_service", lambda settings: service)
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
        assert service.register_bundle.call_args.kwargs["inference_runtime"] == expected


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


def test_server_installs_intake_deadlines_without_automatic_protocol_selection(
    monkeypatch: pytest.MonkeyPatch,
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
    assert captured["limit_concurrency"] == 8
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
