"""Infrastructure settings fail before connecting or exposing secrets."""

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from regimex.service_settings import ServiceSettings


def test_environment_settings_keep_credentials_private(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REGIMEX_DATABASE_DSN", "postgresql://user:private-password@localhost/db")
    monkeypatch.setenv("REGIMEX_DATABASE_SCHEMA", "test_service")
    monkeypatch.setenv("REGIMEX_OBJECT_ROOT", "/tmp/regimex-test-objects")
    monkeypatch.setenv("REGIMEX_HEARTBEAT_SECONDS", "2")
    settings = ServiceSettings.from_env()
    assert settings.database_schema == "test_service"
    assert settings.object_root == Path("/tmp/regimex-test-objects")
    assert settings.heartbeat_seconds == 2
    assert "private-password" not in repr(settings)
    assert "private-password" not in settings.model_dump_json()


def test_missing_connection_fails_with_useful_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REGIMEX_DATABASE_DSN", raising=False)
    with pytest.raises(ValueError, match="REGIMEX_DATABASE_DSN"):
        ServiceSettings.from_env()


def test_private_database_file_is_bounded_and_not_exposed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("REGIMEX_DATABASE_DSN", raising=False)
    path = tmp_path / "database-dsn"
    path.write_text("postgresql://fixture:private-file-password@localhost/service\n")
    path.chmod(0o600)
    monkeypatch.setenv("REGIMEX_DATABASE_DSN_FILE", str(path))
    settings = ServiceSettings.from_env()
    assert "private-file-password" in settings.database_dsn.get_secret_value()
    assert "private-file-password" not in repr(settings)
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        ServiceSettings.from_env()
    path.chmod(0o600)
    path.write_bytes(b"x" * 8193)
    with pytest.raises(ValueError, match="byte limit"):
        ServiceSettings.from_env()


def test_database_file_and_value_cannot_silently_override_each_other(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("REGIMEX_DATABASE_DSN", "dbname=one")
    monkeypatch.setenv("REGIMEX_DATABASE_DSN_FILE", str(tmp_path / "unused"))
    with pytest.raises(ValueError, match="exactly one"):
        ServiceSettings.from_env()


@pytest.mark.parametrize(
    "changes",
    [
        {"database_schema": "public;drop schema public"},
        {"stale_seconds": 20, "heartbeat_seconds": 10},
        {"object_root": Path("objects"), "work_root": Path("objects")},
        {"object_root": Path("objects"), "work_root": Path("objects/work")},
        {"object_root": Path("work/objects"), "work_root": Path("work")},
        {"max_object_bytes": 10, "max_store_bytes": 1},
        {"public_origin": "http://example.invalid"},
        {"public_origin": "http://127.0.0.1:8000"},
        {"public_origin": "https://example.invalid", "allow_loopback_http": True},
        {"public_origin": "https://example.invalid/private/path"},
        {"session_ttl_seconds": 59},
        {"max_sessions_per_token": 0},
        {"analysis_time_budget_seconds": 0},
        {"analysis_memory_budget_bytes": 1},
        {"analysis_scratch_budget_bytes": 0},
    ],
)
def test_invalid_settings_are_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ServiceSettings.model_validate({"database_dsn": SecretStr("dbname=local"), **changes})


def test_analysis_budgets_reach_the_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("psycopg")
    pytest.importorskip("procrastinate")
    from regimex.service_bootstrap import create_service

    monkeypatch.setenv("REGIMEX_DATABASE_DSN", "dbname=not_connected")
    monkeypatch.setenv("REGIMEX_OBJECT_ROOT", str(tmp_path / "objects"))
    monkeypatch.setenv("REGIMEX_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("REGIMEX_ANALYSIS_TIME_BUDGET_SECONDS", "17")
    monkeypatch.setenv("REGIMEX_ANALYSIS_MEMORY_BUDGET_BYTES", str(1024**3))
    monkeypatch.setenv("REGIMEX_ANALYSIS_SCRATCH_BUDGET_BYTES", "123456")
    service = create_service(ServiceSettings.from_env())
    assert service.time_budget_seconds == 17
    assert service.memory_budget_bytes == 1024**3
    assert service.scratch_budget_bytes == 123456


def test_s3_environment_has_explicit_scope_limits_and_external_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("REGIMEX_DATABASE_DSN", "dbname=not_connected")
    monkeypatch.setenv("REGIMEX_OBJECT_BACKEND", "s3")
    monkeypatch.setenv("REGIMEX_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("REGIMEX_S3_ENDPOINT_URL", "https://objects.example.invalid")
    monkeypatch.setenv("REGIMEX_S3_REGION", "us-east-1")
    monkeypatch.setenv("REGIMEX_S3_BUCKET", "private-fixtures")
    monkeypatch.setenv("REGIMEX_S3_PREFIX", "installation-one")
    monkeypatch.setenv("REGIMEX_S3_READ_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "sdk-only-private-key")
    settings = ServiceSettings.from_env()
    assert settings.object_backend == "s3" and settings.s3 is not None
    assert settings.s3.lock_root == tmp_path / "work" / "s3-coordination"
    assert settings.s3.read_timeout_seconds == 7
    assert "sdk-only-private-key" not in settings.model_dump_json()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("REGIMEX_S3_ALLOW_LOOPBACK_HTTP", "perhaps"),
        ("REGIMEX_S3_READ_TIMEOUT_SECONDS", "many"),
        ("REGIMEX_S3_UNRECOGNIZED", "ignored-setting"),
    ],
)
def test_invalid_s3_environment_does_not_silently_fall_back(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv("REGIMEX_DATABASE_DSN", "dbname=not_connected")
    monkeypatch.setenv("REGIMEX_OBJECT_BACKEND", "s3")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="REGIMEX_S3"):
        ServiceSettings.from_env()


def test_s3_settings_require_the_explicit_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REGIMEX_DATABASE_DSN", "dbname=not_connected")
    monkeypatch.delenv("REGIMEX_OBJECT_BACKEND", raising=False)
    monkeypatch.setenv("REGIMEX_S3_BUCKET", "unused-bucket")
    with pytest.raises(ValueError, match="REGIMEX_OBJECT_BACKEND=s3"):
        ServiceSettings.from_env()


def test_invalid_s3_endpoint_does_not_echo_embedded_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REGIMEX_DATABASE_DSN", "dbname=not_connected")
    monkeypatch.setenv("REGIMEX_OBJECT_BACKEND", "s3")
    monkeypatch.setenv("REGIMEX_S3_ENDPOINT_URL", "https://example-user:private-password@localhost")
    monkeypatch.setenv("REGIMEX_S3_REGION", "us-east-1")
    monkeypatch.setenv("REGIMEX_S3_BUCKET", "private-fixtures")
    monkeypatch.setenv("REGIMEX_S3_PREFIX", "private-installation")
    with pytest.raises(ValueError) as failure:
        ServiceSettings.from_env()
    assert "private-password" not in str(failure.value)


def test_browser_origin_and_session_policy_are_explicit_environment_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REGIMEX_DATABASE_DSN", "dbname=not_connected")
    monkeypatch.setenv("REGIMEX_PUBLIC_ORIGIN", "http://127.0.0.1:8000")
    monkeypatch.setenv("REGIMEX_ALLOW_LOOPBACK_HTTP", "true")
    monkeypatch.setenv("REGIMEX_SESSION_TTL_SECONDS", "600")
    monkeypatch.setenv("REGIMEX_MAX_SESSIONS_PER_TOKEN", "3")
    settings = ServiceSettings.from_env()
    assert settings.auth_policy.origin == "http://127.0.0.1:8000"
    assert not settings.auth_policy.secure
    assert settings.auth_policy.session_ttl_seconds == 600
    assert settings.auth_policy.max_sessions_per_token == 3


def test_invalid_browser_transport_option_does_not_fall_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REGIMEX_DATABASE_DSN", "dbname=not_connected")
    monkeypatch.setenv("REGIMEX_ALLOW_LOOPBACK_HTTP", "perhaps")
    with pytest.raises(ValueError, match="REGIMEX_ALLOW_LOOPBACK_HTTP"):
        ServiceSettings.from_env()


def test_invalid_public_origin_does_not_echo_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REGIMEX_DATABASE_DSN", "dbname=not_connected")
    monkeypatch.setenv("REGIMEX_PUBLIC_ORIGIN", "https://example-user:private-secret@localhost")
    with pytest.raises(ValueError) as failure:
        ServiceSettings.from_env()
    assert "private-secret" not in str(failure.value)
