"""Operational entry points remain discoverable in a core-only installation."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

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
            "assert 'register-model' in r.output and 'worker' in r.output; "
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
