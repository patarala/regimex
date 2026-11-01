"""Thin operational commands; optional dependencies load only on execution."""

import asyncio
import ipaddress
import json
import logging
import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

import typer

from regimex.service_settings import ServiceSettings
from regimex.storage_cli import install_storage_commands

app = typer.Typer(
    help="Run and administer the persistent analysis service.", pretty_exceptions_show_locals=False
)


def _execute(action: Callable[[], None]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        settings = ServiceSettings.from_env()
        from regimex.service_logging import configure_service_logging

        configure_service_logging(settings)
        from psycopg import Error as DatabaseError
        from sqlalchemy.exc import SQLAlchemyError

        action()
    except ModuleNotFoundError as exc:
        typer.echo(
            "Install service dependencies with uv sync --locked --extra service; "
            "S3 also requires --extra s3 and computation requires --extra ml.",
            err=True,
        )
        raise typer.Exit(1) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    except (DatabaseError, SQLAlchemyError) as exc:
        # Driver messages and rich tracebacks may contain connection arguments.
        typer.echo("Error: Database infrastructure is unavailable or incompatible.", err=True)
        raise typer.Exit(1) from exc


install_storage_commands(app, _execute)


@app.command("health")
def health(
    component: Annotated[Literal["ready", "models", "worker"], typer.Option()] = "ready",
    queue: Annotated[Literal["analysis", "admin", "maintenance"] | None, typer.Option()] = None,
    port: Annotated[int | None, typer.Option(min=1, max=65535)] = None,
    ca_file: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
    """Probe actual API TLS/readiness or this worker's exact database heartbeat."""

    def action() -> None:
        from regimex.service_health import api_healthy, worker_healthy

        settings = ServiceSettings.from_env()
        if component == "worker":
            if queue is None or port is not None or ca_file is not None:
                raise ValueError("Worker health requires a queue and accepts no HTTP settings")
            healthy = worker_healthy(settings, queue)
        else:
            if queue is not None:
                raise ValueError("An API health probe does not accept a worker queue")
            healthy = api_healthy(settings, component=component, port=port or 8000, ca_file=ca_file)
        typer.echo(
            json.dumps(
                {
                    "status": "healthy" if healthy else "unhealthy",
                    "component": component,
                    "worker_capability": "not_probed",
                }
            )
        )
        if not healthy:
            raise typer.Exit(1)

    _execute(action)


@app.command("migrate")
def migrate() -> None:
    """Apply packaged Alembic and compatible Procrastinate schema migrations."""

    def action() -> None:
        settings = ServiceSettings.from_env()
        from regimex.infrastructure.database import DatabaseRepository

        DatabaseRepository(
            settings.database_dsn.get_secret_value(), schema=settings.database_schema
        ).migrate()
        typer.echo("Database migrations are current.")

    _execute(action)


@app.command("token-create")
def token_create(
    owner: Annotated[str, typer.Option(help="Authenticated local owner ID.")],
    role: Annotated[Literal["user", "admin", "integration"], typer.Option()],
    output: Annotated[Path, typer.Option(help="New private file for the bearer token.")],
    source: Annotated[str | None, typer.Option(help="Required integration source ID.")] = None,
) -> None:
    """Provision a revocable token; persist only its hash in PostgreSQL."""

    def action() -> None:
        settings = ServiceSettings.from_env()
        from regimex.infrastructure.database import DatabaseRepository

        repository = DatabaseRepository(
            settings.database_dsn.get_secret_value(), schema=settings.database_schema
        )
        # Reserve the output before creating credentials: never overwrite an
        # existing token or print a raw token into ordinary process logs.
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            principal, token = repository.issue_token(owner, role, source_namespace=source)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                descriptor = -1
                stream.write(token + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            output.unlink(missing_ok=True)
            raise
        typer.echo(f"Created {principal.role} token {principal.token_id}; saved to {output}.")

    _execute(action)


@app.command("token-revoke")
def token_revoke(token_id: Annotated[UUID, typer.Option()]) -> None:
    """Revoke a provisioned credential by public token ID."""

    def action() -> None:
        settings = ServiceSettings.from_env()
        from regimex.infrastructure.database import DatabaseRepository

        DatabaseRepository(
            settings.database_dsn.get_secret_value(), schema=settings.database_schema
        ).revoke_token(token_id)
        typer.echo("Token revoked.")

    _execute(action)


@app.command("register-model")
def register_model(
    bundle: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    manifest_sha256: Annotated[str, typer.Option()],
    token_file: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    activate: Annotated[
        bool, typer.Option(help="Explicitly activate this verified import.")
    ] = False,
    expected_previous: Annotated[str | None, typer.Option()] = None,
    threads: Annotated[int, typer.Option(min=1, max=256)] = 2,
) -> None:
    """Verify a new-format saved bundle in a computation process and import it."""

    def action() -> None:
        from regimex.model_registry import InferenceRuntime
        from regimex.service_bootstrap import create_service

        service = create_service(ServiceSettings.from_env())
        principal = service.authenticate(token_file.read_text(encoding="utf-8").strip())
        model = service.register_bundle(
            principal,
            bundle,
            manifest_sha256=manifest_sha256,
            inference_runtime=InferenceRuntime(threads=threads),
            activate=activate,
            expected_previous=expected_previous,
        )
        typer.echo(f"Registered {model.model_version_id} ({model.bundle_sha256}).")

    _execute(action)


@app.command("serve")
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8000,
    tls_cert_file: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    tls_key_file: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
    """Serve the API without migrations, model loading, or automatic training."""

    def action() -> None:
        import uvicorn

        from regimex.service_bootstrap import create_app

        settings = ServiceSettings.from_env()
        if (tls_cert_file is None) != (tls_key_file is None):
            raise ValueError("TLS requires both certificate and private key files")
        secure = settings.auth_policy.secure
        if secure != (tls_cert_file is not None):
            raise ValueError(
                "An HTTPS public origin requires direct TLS; "
                "HTTP is limited to explicit loopback development"
            )
        if not secure:
            try:
                loopback = ipaddress.ip_address(host).is_loopback
            except ValueError:
                loopback = False
            if not loopback:
                raise ValueError("HTTP development requires a literal loopback listener")
        if tls_key_file is not None:
            descriptor = os.open(tls_key_file, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            try:
                info = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.geteuid()
                    or info.st_mode & 0o077
                    or info.st_size > 1024**2
                ):
                    raise ValueError(
                        "The TLS private key must be a bounded private file "
                        "owned by the service user"
                    )
            finally:
                os.close(descriptor)
        uvicorn.run(
            create_app(),
            host=host,
            port=port,
            proxy_headers=False,
            access_log=False,
            log_config=None,
            timeout_graceful_shutdown=30,
            limit_concurrency=8,
            ssl_certfile=None if tls_cert_file is None else str(tls_cert_file),
            ssl_keyfile=None if tls_key_file is None else str(tls_key_file),
        )

    _execute(action)


@app.command("activate-model")
def activate_model(
    model_version_id: Annotated[str, typer.Option()],
    token_file: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    expected_previous: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Verify and activate an existing immutable import with compare-and-set."""

    def action() -> None:
        from regimex.service_bootstrap import create_service

        service = create_service(ServiceSettings.from_env())
        principal = service.authenticate(token_file.read_text(encoding="utf-8").strip())
        model = service.activate_model(
            principal, model_version_id, expected_previous=expected_previous
        )
        typer.echo(f"Activated {model.model_version_id} ({model.bundle_sha256}).")

    _execute(action)


@app.command("worker")
def worker(
    queue: Annotated[Literal["analysis", "admin", "maintenance"], typer.Option()] = "analysis",
) -> None:
    """Run one computation at a time, or the separate recovery scheduler."""

    def action() -> None:
        from regimex.service_bootstrap import create_admin_service, create_service
        from regimex.service_worker import create_worker_app

        settings = ServiceSettings.from_env()
        service = create_service(settings)
        if not service.repository.readiness():
            raise ValueError("Apply service migrations before starting workers")
        queue_app = create_worker_app(
            settings,
            service,
            admin_service=create_admin_service(settings, service),
            health_queue=queue,
        )
        from regimex.service_health import worker_health_lease

        with worker_health_lease(settings, queue):
            queue_app.run_worker(
                queues=[queue],
                concurrency=1,
                name=f"regimex-{queue}",
                update_heartbeat_interval=settings.heartbeat_seconds,
                stalled_worker_timeout=settings.stale_seconds,
                shutdown_graceful_timeout=30.0,
            )

    _execute(action)


@app.command("recover")
def recover() -> None:
    """Run one recovery pass; the maintenance worker also schedules this task."""

    def action() -> None:
        from regimex.service_bootstrap import create_service
        from regimex.service_worker import create_worker_app, recover_operations

        settings = ServiceSettings.from_env()
        service = create_service(settings)
        queue_app = create_worker_app(settings, service)

        async def run() -> int:
            async with queue_app.open_async():
                return await recover_operations(settings, service, queue_app)

        typer.echo(f"Recovered {asyncio.run(run())} stale application operations.")

    _execute(action)
