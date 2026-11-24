"""CLI for storage diagnostics, backup, restore and orphan cleanup."""

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
        """Check the object store and report its retained versions and bytes."""

        def action() -> None:
            from regimex.infrastructure.s3_object_store import S3ObjectStore
            from regimex.service_bootstrap import create_object_store

            settings = ServiceSettings.from_env()
            store = create_object_store(settings)
            if isinstance(store, S3ObjectStore):
                counts = store.capacity_report()
            else:
                versions = store.inventory()
                counts = dict(
                    retained_versions=len(versions),
                    retained_payload_bytes=sum(v.size_bytes for v in versions),
                )
            typer.echo(
                json.dumps(
                    {
                        "backend": settings.object_backend,
                        "scope_id": store.scope_id,
                        **counts,
                        "scope": "configuration_and_inventory",
                    },
                    sort_keys=True,
                )
            )

        execute(action)

    @app.command("storage-reconcile-writes")
    def reconcile_writes(
        attestation_file: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
        provider_drained: Annotated[
            bool,
            typer.Option(help="Confirm all previous PUT requests have drained at the provider."),
        ] = False,
        after_key: Annotated[str | None, typer.Option(help="Continue from next_after.")] = None,
    ) -> None:
        """Verify uncertain S3 writes offline and report remaining reservations."""

        def action() -> None:
            from dataclasses import asdict

            from regimex.infrastructure.s3_object_store import S3ObjectStore
            from regimex.service_bootstrap import create_object_store

            store = create_object_store(ServiceSettings.from_env())
            if not isinstance(store, S3ObjectStore):
                raise ValueError(
                    "Write reconciliation requires the PostgreSQL-coordinated S3 store"
                )
            result = store.reconcile_pending_writes(
                _attestation(attestation_file),
                provider_drained=provider_drained,
                after_key=after_key,
            )
            typer.echo(json.dumps(asdict(result)))

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
        after_event_id: Annotated[int, typer.Option(min=0)] = 0,
        limit: Annotated[int, typer.Option(min=1, max=1000)] = 1000,
    ) -> None:
        """Inspect the append-only cleanup audit after completion or interruption."""

        def action() -> None:
            service, principal = _retention(ServiceSettings.from_env(), token_file)
            receipt = service.get_receipt(
                principal, plan_id, after_event_id=after_event_id, limit=limit
            )
            typer.echo(receipt.model_dump_json())

        execute(action)
