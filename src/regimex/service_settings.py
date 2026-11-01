"""Explicit infrastructure configuration; no scientific or server imports."""

import os
import stat
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, SecretStr, ValidationError, model_validator

from regimex.auth_contracts import AuthPolicy
from regimex.contracts import StrictModel
from regimex.infrastructure.s3_config import S3Config

_DEFAULT_WORK_ROOT = Path("var/work")


class ServiceSettings(StrictModel):
    model_config = ConfigDict(hide_input_in_errors=True)
    database_dsn: SecretStr
    database_schema: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,62}$")] = "public"
    object_root: Path = Path("var/objects")
    object_backend: Literal["local", "s3"] = "local"
    s3: S3Config | None = None
    work_root: Path = _DEFAULT_WORK_ROOT
    public_origin: Annotated[str, Field(min_length=1, max_length=2048)] = "https://localhost"
    allow_loopback_http: bool = False
    session_ttl_seconds: Annotated[int, Field(ge=60, le=8 * 3600)] = 1800
    max_sessions_per_token: Annotated[int, Field(ge=1, le=32)] = 8
    max_subjects_per_integration: Annotated[int, Field(ge=1, le=1_000_000)] = 100_000
    max_object_bytes: Annotated[int, Field(ge=1)] = 512 * 1024**2
    max_store_bytes: Annotated[int, Field(ge=1)] = 4 * 1024**3
    max_pending_per_owner: Annotated[int, Field(ge=1, le=1000)] = 8
    cache_budget_bytes: Annotated[int, Field(ge=1)] = 512 * 1024**2
    analysis_time_budget_seconds: Annotated[int, Field(ge=1, le=86400)] = 120
    analysis_memory_budget_bytes: Annotated[int, Field(ge=128 * 1024**2)] = 4 * 1024**3
    analysis_scratch_budget_bytes: Annotated[int, Field(ge=1)] = 128 * 1024**2
    max_admin_pending_per_owner: Annotated[int, Field(ge=1, le=100)] = 4
    max_generation_experiments: Annotated[int, Field(ge=3, le=1000)] = 128
    max_training_seeds: Annotated[int, Field(ge=1, le=64)] = 8
    max_training_epochs: Annotated[int, Field(ge=1, le=1000)] = 1000
    admin_time_budget_seconds: Annotated[int, Field(ge=1, le=86400)] = 3600
    admin_memory_budget_bytes: Annotated[int, Field(ge=128 * 1024**2)] = 4 * 1024**3
    admin_scratch_budget_bytes: Annotated[int, Field(ge=1)] = 4 * 1024**3
    heartbeat_seconds: Annotated[int, Field(ge=1, le=60)] = 10
    stale_seconds: Annotated[int, Field(ge=3, le=86400)] = 120

    @model_validator(mode="after")
    def coherent_limits(self) -> Self:
        _ = self.auth_policy
        if not self.database_dsn.get_secret_value().strip():
            raise ValueError("REGIMEX_DATABASE_DSN is required")
        if self.max_store_bytes < self.max_object_bytes:
            raise ValueError("store capacity must be at least the maximum object size")
        if self.stale_seconds < 3 * self.heartbeat_seconds:
            raise ValueError("stale timeout must allow at least three heartbeats")
        if (self.object_backend == "s3") != (self.s3 is not None):
            raise ValueError("S3 settings are required exactly when the S3 backend is selected")
        if self.s3 is not None and self.max_object_bytes > 512 * 1024**2:
            raise ValueError("the S3 backend supports single-part objects up to 512 MiB")
        object_root, work_root = self.object_root.resolve(), self.work_root.resolve()
        if object_root.is_relative_to(work_root) or work_root.is_relative_to(object_root):
            raise ValueError("object and working roots must not overlap")
        return self

    @property
    def auth_policy(self) -> AuthPolicy:
        return AuthPolicy(
            public_origin=self.public_origin,
            allow_loopback_http=self.allow_loopback_http,
            session_ttl_seconds=self.session_ttl_seconds,
            max_sessions_per_token=self.max_sessions_per_token,
            max_subjects_per_integration=self.max_subjects_per_integration,
        )

    @classmethod
    def from_env(cls) -> Self:
        """Read only named service settings; tokens never belong in task settings."""
        dsn = os.environ.get("REGIMEX_DATABASE_DSN")
        dsn_file = os.environ.get("REGIMEX_DATABASE_DSN_FILE")
        if dsn is not None and dsn_file is not None:
            raise ValueError("Configure exactly one database DSN value or private DSN file")
        if dsn_file is not None:
            descriptor = os.open(dsn_file, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as stream:
                info = os.fstat(stream.fileno())
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.geteuid()
                    or info.st_mode & 0o077
                ):
                    raise ValueError(
                        "The database DSN file must be private and owned by the service user"
                    )
                payload = stream.read(8193)
            if len(payload) > 8192:
                raise ValueError("The database DSN file exceeds its byte limit")
            try:
                dsn = payload.decode("utf-8").strip()
            except UnicodeDecodeError:
                raise ValueError("The database DSN file must contain UTF-8 text") from None
        if not dsn:
            raise ValueError(
                "Set REGIMEX_DATABASE_DSN or REGIMEX_DATABASE_DSN_FILE "
                "before running service commands"
            )
        values: dict[str, object] = {"database_dsn": SecretStr(dsn)}
        for name in (
            "database_schema",
            "object_root",
            "work_root",
            "object_backend",
            "public_origin",
        ):
            value = os.environ.get(f"REGIMEX_{name.upper()}")
            if value is not None:
                values[name] = Path(value) if name.endswith("_root") else value
        allowance = os.environ.get("REGIMEX_ALLOW_LOOPBACK_HTTP")
        if allowance is not None:
            if allowance.lower() not in {"true", "false"}:
                raise ValueError("REGIMEX_ALLOW_LOOPBACK_HTTP must be true or false")
            values["allow_loopback_http"] = allowance.lower() == "true"
        s3_environment = {
            key.removeprefix("REGIMEX_S3_").lower(): value
            for key, value in os.environ.items()
            if key.startswith("REGIMEX_S3_")
        }
        if values.get("object_backend") == "s3":
            values["s3"] = cls._s3_from_env(
                s3_environment, Path(os.environ.get("REGIMEX_WORK_ROOT", str(_DEFAULT_WORK_ROOT)))
            )
        elif s3_environment:
            raise ValueError("REGIMEX_S3 settings require REGIMEX_OBJECT_BACKEND=s3")
        for name in (
            "max_object_bytes",
            "max_store_bytes",
            "max_pending_per_owner",
            "session_ttl_seconds",
            "max_sessions_per_token",
            "max_subjects_per_integration",
            "cache_budget_bytes",
            "analysis_time_budget_seconds",
            "analysis_memory_budget_bytes",
            "analysis_scratch_budget_bytes",
            "max_admin_pending_per_owner",
            "max_generation_experiments",
            "max_training_seeds",
            "max_training_epochs",
            "admin_time_budget_seconds",
            "admin_memory_budget_bytes",
            "admin_scratch_budget_bytes",
            "heartbeat_seconds",
            "stale_seconds",
        ):
            value = os.environ.get(f"REGIMEX_{name.upper()}")
            if value is not None:
                try:
                    values[name] = int(value)
                except ValueError as exc:
                    raise ValueError(f"REGIMEX_{name.upper()} must be an integer") from exc
        return cls.model_validate(values)

    @staticmethod
    def _s3_from_env(environment: dict[str, str], work_root: Path) -> S3Config:
        values: dict[str, object] = {"lock_root": work_root / "s3-coordination"}
        strings = {"endpoint_url", "region", "bucket", "prefix", "credential_profile"}
        paths = {"lock_root", "ca_bundle"}
        integers = {
            "connect_timeout_seconds",
            "read_timeout_seconds",
            "max_attempts",
            "max_inventory_entries",
        }
        for name, value in environment.items():
            if name in strings:
                values[name] = value
            elif name in paths:
                values[name] = Path(value)
            elif name in integers:
                try:
                    values[name] = int(value)
                except ValueError as exc:
                    raise ValueError(f"REGIMEX_S3_{name.upper()} must be an integer") from exc
            elif name == "allow_loopback_http":
                if value.lower() not in {"true", "false"}:
                    raise ValueError("REGIMEX_S3_ALLOW_LOOPBACK_HTTP must be true or false")
                values[name] = value.lower() == "true"
            else:
                raise ValueError(f"Unknown S3 setting: REGIMEX_S3_{name.upper()}")
        try:
            return S3Config.model_validate(values)
        except ValidationError as exc:
            raise ValueError(
                "Invalid S3 endpoint, namespace, TLS or capacity configuration"
            ) from exc
