"""Operator commands for platform connections and their narrowly scoped keys."""

import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated
from uuid import UUID

import typer

from regimex.service_settings import ServiceSettings

if TYPE_CHECKING:
    from regimex.infrastructure.integration_repository import IntegrationRepository
    from regimex.service_contracts import Principal


def _administrator(context: typer.Context) -> tuple[IntegrationRepository, Principal]:
    from regimex.infrastructure.database import DatabaseRepository
    from regimex.infrastructure.integration_repository import IntegrationRepository

    path = context.obj
    if not isinstance(path, Path):
        raise ValueError("Supply --token-file before the connection command.")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ValueError("The administrator token file must be private and owned by this user.")
        raw = stream.read(1025)
    if len(raw) > 1024:
        raise ValueError("The administrator token file exceeds its size limit.")
    settings = ServiceSettings.from_env()
    database = DatabaseRepository(
        settings.database_dsn.get_secret_value(), schema=settings.database_schema
    )
    principal = database.authenticate(raw.decode("ascii").strip())
    database._admin(principal)
    return IntegrationRepository(database), principal


def install_integration_commands(
    app: typer.Typer, execute: Callable[[Callable[[], None]], None]
) -> None:
    connections = typer.Typer(help="Manage platform connections and API keys.")
    app.add_typer(connections, name="connections")

    @connections.callback()
    def credentials(
        context: typer.Context,
        token_file: Annotated[
            Path | None, typer.Option(help="Private administrator token file.")
        ] = None,
    ) -> None:
        context.obj = token_file

    @connections.command("list")
    def list_connections(
        context: typer.Context,
        limit: Annotated[int, typer.Option(min=1, max=100)] = 50,
        offset: Annotated[int, typer.Option(min=0, max=1_000_000)] = 0,
    ) -> None:
        """List connections without exposing any credentials."""

        def action() -> None:
            repository, actor = _administrator(context)
            for item in repository.list(actor, limit=limit, offset=offset):
                typer.echo(item.model_dump_json())

        execute(action)

    @connections.command("create")
    def create_connection(
        context: typer.Context,
        owner: Annotated[str, typer.Option()],
        source: Annotated[str, typer.Option()],
        name: Annotated[str, typer.Option()],
    ) -> None:
        """Register a stable integration account and source namespace."""

        def action() -> None:
            from regimex.access_contracts import IntegrationCreate

            repository, actor = _administrator(context)
            item = repository.create(
                actor,
                IntegrationCreate(owner_id=owner, source_namespace=source, display_name=name),
            )
            typer.echo(item.model_dump_json())

        execute(action)

    @connections.command("update")
    def update_connection(
        context: typer.Context,
        integration_id: UUID,
        revision: Annotated[int, typer.Option(min=0)],
        name: Annotated[str | None, typer.Option()] = None,
        enabled: Annotated[bool | None, typer.Option("--enable/--disable")] = None,
    ) -> None:
        """Rename or suspend a connection using its last observed revision."""

        def action() -> None:
            from regimex.access_contracts import IntegrationUpdate

            repository, actor = _administrator(context)
            item = repository.update(
                actor,
                integration_id,
                IntegrationUpdate(expected_revision=revision, display_name=name, enabled=enabled),
            )
            typer.echo(item.model_dump_json())

        execute(action)

    @connections.command("keys")
    def list_keys(
        context: typer.Context,
        integration_id: UUID,
        limit: Annotated[int, typer.Option(min=1, max=100)] = 50,
        offset: Annotated[int, typer.Option(min=0, max=1_000_000)] = 0,
    ) -> None:
        """List key metadata; a saved secret cannot be retrieved."""

        def action() -> None:
            repository, actor = _administrator(context)
            for item in repository.list_keys(actor, integration_id, limit=limit, offset=offset):
                typer.echo(item.model_dump_json())

        execute(action)

    @connections.command("key-create")
    def create_key(
        context: typer.Context,
        integration_id: UUID,
        name: Annotated[str, typer.Option()],
        output: Annotated[Path, typer.Option(help="New private file for the one-time secret.")],
        days: Annotated[int, typer.Option(min=1, max=365)] = 30,
        scope: Annotated[
            list[str] | None,
            typer.Option(help="Repeat read, upload, analyze or delete; deletion is opt-in."),
        ] = None,
    ) -> None:
        """Issue a key with an expiry and explicit capabilities."""

        def action() -> None:
            from regimex.access_contracts import IntegrationKeyCreate

            repository, actor = _administrator(context)
            request = IntegrationKeyCreate.model_validate(
                {
                    "name": name,
                    "lifetime_days": days,
                    "scopes": tuple(scope) if scope is not None else ("read", "upload", "analyze"),
                }
            )
            descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            issued = None
            try:
                issued = repository.issue_key(actor, integration_id, request)
                with os.fdopen(descriptor, "w", encoding="ascii") as stream:
                    descriptor = -1
                    stream.write(issued.token.get_secret_value() + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            except BaseException:
                if descriptor >= 0:
                    os.close(descriptor)
                output.unlink(missing_ok=True)
                if issued is not None:
                    repository.revoke_key(actor, integration_id, issued.key.token_id)
                raise
            typer.echo(issued.key.model_dump_json())

        execute(action)

    @connections.command("key-revoke")
    def revoke_key(context: typer.Context, integration_id: UUID, token_id: UUID) -> None:
        """Permanently revoke this key without removing saved analyses."""

        def action() -> None:
            repository, actor = _administrator(context)
            repository.revoke_key(actor, integration_id, token_id)
            typer.echo("Key revoked.")

        execute(action)

    @connections.command("events")
    def list_events(
        context: typer.Context,
        integration_id: UUID,
        limit: Annotated[int, typer.Option(min=1, max=100)] = 50,
        offset: Annotated[int, typer.Option(min=0, max=1_000_000)] = 0,
    ) -> None:
        """Inspect access changes without printing secrets or input files."""

        def action() -> None:
            repository, actor = _administrator(context)
            for item in repository.events(actor, integration_id, limit=limit, offset=offset):
                typer.echo(item.model_dump_json())

        execute(action)
