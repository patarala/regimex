"""Real TCP gateway -> authenticated Regimex -> PostgreSQL -> saved baseline/report."""

from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import pytest
from pydantic import SecretBytes, SecretStr
from support.simulator_gateway.client import GatewayError, GatewaySettings, RegimexClient
from support.simulator_gateway.jwt import SimulatorJWT, json_bytes

from regimex.exports import ExportPayloadV2, ExportV2, seal_export

if TYPE_CHECKING:
    from fastapi import FastAPI
    from uvicorn import Server

    from regimex.application.service import AnalysisService
    from regimex.infrastructure.auth_repository import AuthRepository


@dataclass
class RunningServer:
    origin: str
    server: Server
    thread: threading.Thread

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)
        assert not self.thread.is_alive()


@contextmanager
def tcp_server(factory: Callable[[str], FastAPI]) -> Iterator[RunningServer]:
    import uvicorn

    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    origin = f"http://127.0.0.1:{listener.getsockname()[1]}"
    server = uvicorn.Server(
        uvicorn.Config(
            factory(origin),
            host="127.0.0.1",
            log_level="warning",
            access_log=False,
            lifespan="on",
            timeout_graceful_shutdown=2,
        )
    )
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    running = RunningServer(origin, server, thread)
    try:
        deadline = time.monotonic() + 10
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started and thread.is_alive()
        yield running
    finally:
        running.stop()
        listener.close()


@dataclass
class PersistentGateway:
    service: AnalysisService
    auth_repository: AuthRepository
    integration_token: SecretStr
    export: ExportV2
    rerun: ExportV2
    bob_export: ExportV2


@pytest.fixture
def persistent_gateway(tmp_path: Path) -> Iterator[PersistentGateway]:
    for dependency in (
        "fastapi",
        "uvicorn",
        "httpx2",
        "psycopg",
        "procrastinate",
        "alembic",
        "numpy",
        "sklearn",
    ):
        pytest.importorskip(dependency)
    import psycopg
    from test_application_training import source
    from test_baseline_backend import bundle

    from regimex.application.service import AnalysisService
    from regimex.infrastructure.auth_repository import AuthRepository
    from regimex.infrastructure.database import DatabaseRepository
    from regimex.infrastructure.object_store import LocalObjectStore
    from regimex.model_registry import ArchitectureRegistry, InferenceRuntime, baseline_descriptor

    dsn = os.environ.get("REGIMEX_TEST_DATABASE_DSN")
    if not dsn:
        pytest.skip("set REGIMEX_TEST_DATABASE_DSN to a disposable PostgreSQL database")
    schema = "regimex_gateway_" + uuid4().hex
    repository = DatabaseRepository(dsn, schema=schema)
    repository.migrate()
    try:
        with repository.queue_app.open():
            service = AnalysisService(
                repository,
                LocalObjectStore(
                    tmp_path / "objects",
                    max_object_bytes=32 * 1024**2,
                    max_total_bytes=128 * 1024**2,
                ),
                ArchitectureRegistry((baseline_descriptor(),)),
                tmp_path / "work",
            )
            admin = repository.authenticate(repository.create_token("fixture-admin", "admin"))
            _, _, _, checksum = bundle(tmp_path)
            service.register_bundle(
                admin,
                tmp_path / "bundle",
                manifest_sha256=checksum,
                inference_runtime=InferenceRuntime(threads=1),
                activate=True,
            )
            token = repository.create_token(
                "fixture-simulator-backend",
                "integration",
                source_namespace="test-simulator",
            )
            export, rerun, bob = source(8101).export, source(8102).export, source(8103).export
            rerun_value = rerun.model_dump(mode="json", exclude={"content_sha256"})
            assert isinstance(rerun_value["source"], dict)
            rerun_value["source"]["configuration_id"] = export.source.configuration_id
            rerun = seal_export(ExportPayloadV2.model_validate_json(json_bytes(rerun_value)))
            yield PersistentGateway(
                service, AuthRepository(repository), SecretStr(token), export, rerun, bob
            )
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(
                psycopg.sql.SQL("DROP SCHEMA {} CASCADE").format(
                    psycopg.sql.Identifier(schema),
                )
            )
        # Predictors are cached in read-only private directories during use.
        for path, directories, files in os.walk(tmp_path / "work"):
            Path(path).chmod(0o700)
            for name in directories:
                (Path(path) / name).chmod(0o700)
            for name in files:
                (Path(path) / name).chmod(0o600)


def test_gateway_real_tcp_restart_scope_result_report_and_retry(
    persistent_gateway: PersistentGateway,
) -> None:
    import httpx2
    from support.simulator_gateway.app import FixtureCatalog, FixtureRun, create_gateway

    from regimex.api import create_app
    from regimex.application.auth import AuthService
    from regimex.auth_contracts import AuthPolicy

    context = persistent_gateway
    jwt = SimulatorJWT(
        "contract-simulator",
        "contract-gateway",
        SecretBytes(b"public-deliberate-fixture-key-0123456789"),
    )
    catalog = FixtureCatalog("test-simulator")
    for owner, export in (
        ("alice", context.export),
        ("alice", context.rerun),
        ("bob", context.bob_export),
    ):
        catalog.register(
            FixtureRun.completed(
                export.source.configuration_id,
                owner,
                export,
                results={"trades": [1, 2], "owner": owner},
            )
        )

    def regimex_app(origin: str) -> FastAPI:
        auth = AuthService(
            context.auth_repository, AuthPolicy(public_origin=origin, allow_loopback_http=True)
        )
        return create_app(context.service, auth_service=auth)

    with tcp_server(regimex_app) as upstream:
        settings = GatewaySettings(
            source_namespace="test-simulator",
            regimex_origin=upstream.origin,
            integration_token=context.integration_token,
            allow_loopback_http=True,
        )
        simulation_id, run_id = context.export.source.configuration_id, context.export.source.run_id
        base = f"/api/simulations/{simulation_id}/regime-analyses"
        headers = {"Authorization": "Bearer " + jwt.issue("alice")}
        with (
            tcp_server(lambda origin: create_gateway(settings, jwt, catalog)) as gateway,
            httpx2.Client(base_url=gateway.origin, timeout=10, trust_env=False) as browser,
        ):
            created = browser.post(base, json={"run_id": run_id}, headers=headers)
            assert created.status_code == 202, created.text
            analysis_id = UUID(created.json()["analysis_id"])
            retry = browser.post(base, json={"run_id": run_id}, headers=headers)
            assert retry.status_code == 202 and retry.json()["analysis_id"] == str(analysis_id)
            assert created.json()["state"] == "queued"
            assert created.headers["location"].startswith(base + "/")
            assert browser.get(created.headers["location"], headers=headers).status_code == 200
            unavailable = browser.get(created.json()["report_url"], headers=headers)
            assert unavailable.status_code == 409

            principal = context.auth_repository.bind_subject(
                context.integration_token.get_secret_value(),
                "alice",
                run_id,
            )
            record = context.service.get_analysis(principal, analysis_id)
            completed = context.service.run_analysis(record.operation.operation_id)
            assert completed is not None and completed.state == "succeeded"
            status = browser.get(created.headers["location"], headers=headers)
            assert status.status_code == 200 and status.json()["result_available"] is True
            assert status.json()["report_available"] is True
            result = browser.get(status.json()["result_url"], headers=headers)
            report = browser.get(status.json()["report_url"], headers=headers)
            assert result.status_code == 200 and result.json()["source"]["run_id"] == run_id
            assert result.json()["snapshot_sha256"] == context.export.content_sha256
            assert report.status_code == 200 and report.headers["content-type"].startswith(
                "text/html"
            )
            assert report.headers["content-disposition"].endswith(f'{analysis_id}.html"')
            assert report.headers["content-security-policy"] == "sandbox"
            assert b"<!doctype html" in report.content.lower()
            expected_result, expected_report = result.content, report.content
            wrong_run = browser.get(
                f"{base}/{analysis_id}",
                params={"run_id": context.rerun.source.run_id},
                headers=headers,
            )
            assert wrong_run.status_code == 404
            bob = browser.get(
                f"/api/simulations/{context.bob_export.source.configuration_id}/regime-analyses/{analysis_id}",
                params={"run_id": context.bob_export.source.run_id},
                headers={"Authorization": "Bearer " + jwt.issue("bob")},
            )
            assert bob.status_code == 404

        # A fresh gateway process requires no analysis->run cache. The backend
        # credential can rotate while its account/source subject binding persists.
        rotated = context.service.repository.create_token(
            "fixture-simulator-backend",
            "integration",
            source_namespace="test-simulator",
        )
        settings = settings.model_copy(update={"integration_token": SecretStr(rotated)})
        with (
            tcp_server(lambda origin: create_gateway(settings, jwt, catalog)) as gateway,
            httpx2.Client(base_url=gateway.origin, timeout=10, trust_env=False) as browser,
        ):
            replay = browser.post(base, json={"run_id": run_id}, headers=headers)
            assert replay.status_code == 202 and replay.json()["analysis_id"] == str(analysis_id)
            assert (
                browser.get(replay.json()["result_url"], headers=headers).content == expected_result
            )
            assert (
                browser.get(replay.json()["report_url"], headers=headers).content == expected_report
            )
            with context.service.repository.connection() as connection:
                assert connection.execute("SELECT count(*) FROM regimex_analyses").fetchone() == (
                    1,
                )
            isolated_token = context.service.repository.create_token(
                "another-simulator-backend",
                "integration",
                source_namespace="test-simulator",
            )
            isolated = RegimexClient(
                settings.model_copy(
                    update={
                        "integration_token": SecretStr(isolated_token),
                    }
                )
            )
            with pytest.raises(GatewayError) as caught:
                isolated.json(
                    "GET",
                    f"/api/v1/integration/analyses/{analysis_id}",
                    subject="alice",
                    run_id=run_id,
                )
            assert caught.value.status == 404
            upstream.stop()
            failed = browser.post(base, json={"run_id": run_id}, headers=headers)
            assert failed.status_code == 503 and failed.json()["error"]["retryable"] is True
            original = browser.get(
                f"/api/simulations/{simulation_id}/results",
                params={"run_id": run_id},
                headers=headers,
            )
            assert original.status_code == 200 and original.json()["trades"] == [1, 2]
