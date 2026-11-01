"""Construct explicitly configured service components without changing the schema."""

from typing import TYPE_CHECKING

from regimex.service_settings import ServiceSettings

if TYPE_CHECKING:
    from fastapi import FastAPI

    from regimex.application.admin_service import AdminService
    from regimex.application.benchmark_service import BenchmarkService
    from regimex.application.service import AnalysisService
    from regimex.infrastructure.admin_repository import AdminRepository
    from regimex.infrastructure.object_store import LocalObjectStore
    from regimex.infrastructure.s3_object_store import S3ObjectStore


def create_object_store(settings: ServiceSettings) -> LocalObjectStore | S3ObjectStore:
    if settings.s3 is not None:
        from regimex.infrastructure.s3_object_store import S3ObjectStore

        return S3ObjectStore(
            settings.s3,
            max_object_bytes=settings.max_object_bytes,
            max_total_bytes=settings.max_store_bytes,
        )
    from regimex.infrastructure.object_store import LocalObjectStore

    return LocalObjectStore(
        settings.object_root,
        max_object_bytes=settings.max_object_bytes,
        max_total_bytes=settings.max_store_bytes,
    )


def create_service(settings: ServiceSettings) -> AnalysisService:
    from regimex.application.service import AnalysisService
    from regimex.infrastructure.database import DatabaseRepository
    from regimex.model_registry import ArchitectureRegistry

    return AnalysisService(
        DatabaseRepository(
            settings.database_dsn.get_secret_value(), schema=settings.database_schema
        ),
        create_object_store(settings),
        ArchitectureRegistry.discover(),
        settings.work_root,
        max_pending=settings.max_pending_per_owner,
        heartbeat_seconds=settings.heartbeat_seconds,
        cache_budget_bytes=settings.cache_budget_bytes,
        time_budget_seconds=settings.analysis_time_budget_seconds,
        memory_budget_bytes=settings.analysis_memory_budget_bytes,
        scratch_budget_bytes=settings.analysis_scratch_budget_bytes,
    )


def create_admin_service(
    settings: ServiceSettings, service: AnalysisService | None = None
) -> AdminService:
    from regimex.application.admin_service import AdminService
    from regimex.infrastructure.admin_repository import AdminRepository

    analysis = service or create_service(settings)
    repository = AdminRepository(analysis.repository)
    benchmark = create_benchmark_service(analysis, repository)
    return AdminService(
        analysis,
        repository,
        settings,
        authorize_selection=benchmark.authorize_selection,
        authorize_final_evaluation=benchmark.authorize_final_evaluation,
    )


def create_benchmark_service(
    analysis: AnalysisService, administration: AdminRepository | None = None
) -> BenchmarkService:
    from regimex.application.benchmark_service import BenchmarkService
    from regimex.infrastructure.admin_repository import AdminRepository

    return BenchmarkService(
        analysis.repository,
        administration or AdminRepository(analysis.repository),
        analysis.store,
        analysis.work_root,
    )


def create_app() -> FastAPI:
    """Uvicorn factory: construction is lightweight and never auto-migrates."""
    from regimex.api import create_app as api_factory
    from regimex.application.auth import AuthService
    from regimex.infrastructure.auth_repository import AuthRepository

    settings = ServiceSettings.from_env()
    analysis = create_service(settings)
    administration = create_admin_service(settings, analysis)
    app = api_factory(
        analysis,
        admin_service=administration,
        benchmark_service=create_benchmark_service(analysis, administration.repository),
        auth_service=AuthService(AuthRepository(analysis.repository), settings.auth_policy),
    )

    from regimex.service_logging import install_http_logging

    install_http_logging(app)
    return app
