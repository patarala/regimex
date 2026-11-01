"""Explicit storage checks and offline maintenance over application use cases."""

import json
import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated
from uuid import UUID

import typer

from regimex.service_settings import ServiceSettings

if TYPE_CHECKING:
    from regimex.application.retention import RetentionService
    from regimex.infrastructure.maintenance import ServicesStoppedAttestation
    from regimex.service_contracts import Principal


def _read_regular(path: Path, maximum: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum:
            raise ValueError("Maintenance inputs must be bounded regular files.")
        data = stream.read(maximum + 1)
        if len(data) > maximum:
            raise ValueError("Maintenance input exceeded its byte limit.")
        return data


def _attestation(path: Path) -> ServicesStoppedAttestation:
    from regimex.infrastructure.maintenance import ServicesStoppedAttestation

    return ServicesStoppedAttestation.model_validate_json(_read_regular(path, 16 * 1024))


def _retention(settings: ServiceSettings, token_file: Path) -> tuple[RetentionService, Principal]:
    from regimex.application.retention import RetentionService
    from regimex.infrastructure.database import DatabaseRepository
    from regimex.infrastructure.retention_repository import RetentionRepository
    from regimex.service_bootstrap import create_object_store

    database = DatabaseRepository(
        settings.database_dsn.get_secret_value(), schema=settings.database_schema
    )
    principal = database.authenticate(_read_regular(token_file, 4096).decode().strip())
    return (
        RetentionService(RetentionRepository(database), create_object_store(settings)),
        principal,
    )


def install_storage_commands(
    app: typer.Typer, execute: Callable[[Callable[[], None]], None]
) -> None:
    @app.command("storage-check")
    def storage_check() -> None:
        """Probe storage explicitly; this does not establish model or worker readiness."""

        def action() -> None:
            from regimex.infrastructure.s3_object_store import S3ObjectStore
            from regimex.service_bootstrap import create_object_store

            settings = ServiceSettings.from_env()
            store = create_object_store(settings)
            if isinstance(store, S3ObjectStore):
                store.check()
            versions = store.inventory()
            typer.echo(
                json.dumps(
                    {
                        "backend": settings.object_backend,
                        "scope_id": store.scope_id,
                        "retained_versions": len(versions),
                        "retained_payload_bytes": sum(item.size_bytes for item in versions),
                        "scope": "configuration_and_inventory",
                    },
                    sort_keys=True,
                )
            )

        execute(action)

    @app.command("backup")
    def backup(
        output: Annotated[Path, typer.Option(help="New private backup directory.")],
        attestation_file: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
        pg_dump: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
        pg_restore: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    ) -> None:
        """Back up the stopped installation, including every registered object."""

        def action() -> None:
            from regimex.application.recovery import backup_service
            from regimex.recovery_contracts import PostgresTools
            from regimex.service_bootstrap import create_object_store

            settings = ServiceSettings.from_env()
            result = backup_service(
                settings.database_dsn,
                settings.database_schema,
                create_object_store(settings),
                output,
                tools=PostgresTools.discover(pg_dump=pg_dump, pg_restore=pg_restore),
                attestation=_attestation(attestation_file),
            )
            typer.echo(
                f"Backup {result.manifest.backup_id}: {len(result.manifest.objects)} objects. "
                f"Manifest SHA-256: {result.manifest_sha256}"
            )

        execute(action)

    @app.command("restore")
    def restore(
        input_path: Annotated[
            Path, typer.Option("--input", exists=True, file_okay=False, help="Verified backup.")
        ],
        manifest_sha256: Annotated[str, typer.Option(help="Digest recorded outside this backup.")],
        attestation_file: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
        pg_dump: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
        pg_restore: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    ) -> None:
        """Restore only into an empty database and fresh object namespace, offline."""

        def action() -> None:
            from regimex.application.recovery import restore_service
            from regimex.recovery_contracts import PostgresTools
            from regimex.service_bootstrap import create_object_store

            settings = ServiceSettings.from_env()
            store = create_object_store(settings)
            receipt = restore_service(
                settings.database_dsn,
                store,
                input_path,
                expected_manifest_sha256=manifest_sha256,
                assert_empty_store=store.assert_empty_store,
                tools=PostgresTools.discover(pg_dump=pg_dump, pg_restore=pg_restore),
                attestation=_attestation(attestation_file),
            )
            typer.echo(
                f"Restored backup {receipt.backup_id}: {receipt.object_count} objects, "
                f"{receipt.changed_versions} rebound object generations."
            )

        execute(action)

    @app.command("storage-gc-plan")
    def gc_plan(
        output: Annotated[Path, typer.Option(help="New private review-plan JSON file.")],
        token_file: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
        attestation_file: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
        grace_seconds: Annotated[int, typer.Option(min=86400)] = 86400,
    ) -> None:
        """Record a bounded orphan plan; preserve every registered artifact."""

        def action() -> None:
            service, principal = _retention(ServiceSettings.from_env(), token_file)
            plan = service.plan(
                principal,
                attestation=_attestation(attestation_file),
                grace_seconds=grace_seconds,
            )
            service.export_plan(principal, plan.plan_id, output)
            typer.echo(
                f"Plan {plan.plan_id}: {len(plan.objects)} object versions, "
                f"{len(plan.multipart)} incomplete uploads. Review SHA-256: {plan.sha256}"
            )

        execute(action)

    @app.command("storage-gc-apply")
    def gc_apply(
        plan_id: Annotated[UUID, typer.Option()],
        plan_sha256: Annotated[str, typer.Option(help="Digest of the reviewed persisted plan.")],
        token_file: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
        attestation_file: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    ) -> None:
        """Fence and remove reviewed orphan versions while services are stopped."""

        def action() -> None:
            service, principal = _retention(ServiceSettings.from_env(), token_file)
            receipt = service.apply(
                principal,
                plan_id,
                expected_sha256=plan_sha256,
                attestation=_attestation(attestation_file),
            )
            typer.echo(receipt.model_dump_json())
            if not receipt.complete:
                raise RuntimeError("Cleanup is incomplete; inspect the persisted receipt.")

        execute(action)

    @app.command("storage-gc-receipt")
    def gc_receipt(
        plan_id: Annotated[UUID, typer.Option()],
        token_file: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    ) -> None:
        """Inspect the append-only cleanup audit after completion or interruption."""

        def action() -> None:
            service, principal = _retention(ServiceSettings.from_env(), token_file)
            typer.echo(service.get_receipt(principal, plan_id).model_dump_json())

        execute(action)
