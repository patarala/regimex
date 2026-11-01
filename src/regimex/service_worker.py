"""Queue entry points; operations, rather than deliveries, own result state."""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal
from uuid import UUID

import procrastinate
import psycopg

from regimex.application.service import AnalysisService
from regimex.service_settings import ServiceSettings

if TYPE_CHECKING:
    from regimex.application.admin_service import AdminService


def create_worker_app(
    settings: ServiceSettings,
    service: AnalysisService,
    *,
    admin_service: AdminService | None = None,
    health_queue: Literal["analysis", "admin", "maintenance"] | None = None,
) -> procrastinate.App:
    connector = procrastinate.PsycopgConnector(
        conninfo=settings.database_dsn.get_secret_value(),
        kwargs={
            "application_name": "regimex",
            "options": f"-csearch_path={settings.database_schema} -cstatement_timeout=10000",
            "connect_timeout": 5,
        },
        min_size=1,
        max_size=4,
        timeout=5,
    )
    app = procrastinate.App(connector=connector)
    if health_queue is not None:
        from regimex.service_health import TrackedJobManager

        app.job_manager = TrackedJobManager(
            connector=connector, settings=settings, queue=health_queue
        )

    @app.task(
        name="regimex.analysis",
        queue="analysis",
        retry=procrastinate.RetryStrategy(
            max_attempts=3,
            exponential_wait=2,
            retry_exceptions=(
                psycopg.OperationalError,
                psycopg.InterfaceError,
                psycopg.errors.SerializationFailure,
                psycopg.errors.DeadlockDetected,
                OSError,
            ),
        ),
    )
    def analyze(operation_id: str) -> None:
        service.run_analysis(UUID(operation_id))

    if admin_service is not None:

        @app.task(
            name="regimex.admin",
            queue="admin",
            retry=procrastinate.RetryStrategy(
                max_attempts=3,
                exponential_wait=2,
                retry_exceptions=(
                    psycopg.OperationalError,
                    psycopg.InterfaceError,
                    psycopg.errors.SerializationFailure,
                    psycopg.errors.DeadlockDetected,
                    OSError,
                ),
            ),
        )
        def administer(operation_id: str) -> None:
            admin_service.run_admin(UUID(operation_id))

    @app.periodic(cron="* * * * *")
    @app.task(name="regimex.recover", queue="maintenance")
    async def recover(timestamp: int) -> None:
        # The timestamp identifies a scheduled delivery, never an observation time.
        await recover_operations(settings, service, app)

    return app


async def recover_operations(
    settings: ServiceSettings, service: AnalysisService, app: procrastinate.App
) -> int:
    """Fence stale application attempts, then redeliver abandoned queue tasks."""
    cutoff = datetime.now(UTC) - timedelta(seconds=settings.stale_seconds)
    recovered = await asyncio.to_thread(service.repository.recover_stale, cutoff)
    queued = await asyncio.to_thread(service.repository.recover_queued_deliveries)
    stalled = await app.job_manager.get_stalled_jobs(seconds_since_heartbeat=settings.stale_seconds)
    for job in stalled:
        if job.task_name in {"regimex.analysis", "regimex.admin", "regimex.recover"}:
            # A delivery abandoned after result commit runs a terminal no-op. A
            # duplicate abandoned before commit must claim the fenced operation.
            await app.job_manager.retry_job(job)
    return len(recovered) + len(queued)
