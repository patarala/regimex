"""The real temporal backend uses shared candidate and saved-analysis workflows."""

import asyncio
import os
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest
from pydantic import SecretStr

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("sklearn")

from test_application_training import PROFILE, prepared_data, source

from regimex.application.service import AnalysisService
from regimex.application.training import (
    SharedSelectionEvaluator,
    TrainingRequest,
    observation_inputs,
    train_candidate,
)
from regimex.backends.tcn import TCNBackend
from regimex.infrastructure.object_store import LocalObjectStore
from regimex.model_registry import (
    ArchitectureRegistry,
    InferenceRuntime,
    TrainingPolicy,
    TrainingRuntime,
    baseline_descriptor,
    tcn_descriptor,
)
from regimex.service_settings import ServiceSettings


def request() -> TrainingRequest:
    return TrainingRequest(
        run_id="tcn-application-fixture",
        model_version_id="candidate-tcn",
        architecture_id="tcn",
        parameters={"channels": 4, "dilations": [1, 2]},
        policy=TrainingPolicy(
            optimizer="adamw", seeds=(11,), epochs=3, batch_size=1, validation_interval=1
        ),
        training_runtime=TrainingRuntime(threads=4),
        verification_runtime=InferenceRuntime(threads=2, max_batch_size=1),
        analysis_profile_id="fixture-profile",
        build_id="test-build",
    )


def registry() -> ArchitectureRegistry:
    return ArchitectureRegistry((baseline_descriptor(), tcn_descriptor()))


def test_tcn_uses_unchanged_shared_training_and_decoded_selection(tmp_path: Path) -> None:
    data, _ = prepared_data()
    root = tmp_path / "candidate"
    result = train_candidate(data, request(), root, registry=registry(), worker_id="test-tcn")
    assert result.architecture_id == "tcn" and result.quality_status == "validation_only"
    assert result.training_runtime.threads == 4 and result.verification_runtime.threads == 2
    predictor = TCNBackend().load_bundle(
        root, manifest_sha256=result.manifest_sha256, runtime=request().verification_runtime
    )
    validation = observation_inputs(data.validation)
    evaluator = SharedSelectionEvaluator(
        validation,
        (data.validation[0].annotations.intervals,),
        PROFILE,
        "human",
    )
    assert evaluator(predictor.predict(validation, runtime=request().verification_runtime)) == (
        result.validation_segment_macro_f1
    )


def test_tcn_candidate_runs_through_real_persistent_queue_without_retraining(
    tmp_path: Path,
) -> None:
    dsn = os.environ.get("REGIMEX_TEST_DATABASE_DSN")
    if not dsn:
        pytest.skip("set REGIMEX_TEST_DATABASE_DSN to a disposable PostgreSQL database")
    psycopg = pytest.importorskip("psycopg")
    pytest.importorskip("alembic")
    pytest.importorskip("procrastinate")
    from regimex.infrastructure.database import DatabaseRepository
    from regimex.service_worker import create_worker_app

    data, _ = prepared_data()
    candidate = train_candidate(
        data, request(), tmp_path / "candidate", registry=registry(), worker_id="test-tcn"
    )
    schema = "regimex_tcn_workflow_" + uuid4().hex
    repository = DatabaseRepository(dsn, schema=schema)
    settings = ServiceSettings(
        database_dsn=SecretStr(dsn),
        database_schema=schema,
        object_root=tmp_path / "objects",
        work_root=tmp_path / "work",
    )
    service = AnalysisService(
        repository,
        LocalObjectStore(
            settings.object_root,
            max_object_bytes=settings.max_object_bytes,
            max_total_bytes=settings.max_store_bytes,
        ),
        registry(),
        settings.work_root,
        heartbeat_seconds=0.05,
    )
    try:
        repository.migrate()
        administrator, _ = repository.issue_token("administrator", "admin")
        owner, _ = repository.issue_token("owner", "user")
        service.register_bundle(
            administrator,
            tmp_path / "candidate",
            manifest_sha256=candidate.manifest_sha256,
            inference_runtime=InferenceRuntime(threads=2),
            activate=True,
        )
        snapshot = service.create_snapshot(owner, source(9000).export)
        queued = service.submit_analysis(
            owner, snapshot.snapshot_id, idempotency_key="tcn-analysis"
        )

        async def drain() -> None:
            app = create_worker_app(settings, service)
            async with app.open_async():
                await asyncio.wait_for(
                    app.run_worker_async(
                        queues=["analysis"],
                        concurrency=1,
                        wait=False,
                        listen_notify=False,
                        install_signal_handlers=False,
                        shutdown_graceful_timeout=1.0,
                    ),
                    timeout=30.0,
                )

        asyncio.run(drain())
        result = service.get_result(owner, queued.analysis_id)
        assert result["architecture_id"] == "tcn"
        assert result["source"] == source(9000).export.source.model_dump(mode="json")
        sequences = cast(list[dict[str, object]], result["sequences"])
        assert len(cast(list[object], sequences[0]["probabilities"])) == 192
        assert service.run_analysis(queued.analysis_id) is None
        with repository.connection() as connection:
            assert connection.execute(
                "SELECT status FROM procrastinate_jobs WHERE task_name='regimex.analysis'"
            ).fetchone() == ("succeeded",)
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(
                psycopg.sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    psycopg.sql.Identifier(schema)
                )
            )
        for parent, directories, files in os.walk(settings.work_root):
            Path(parent).chmod(0o700)
            for name in directories:
                (Path(parent) / name).chmod(0o700)
            for name in files:
                (Path(parent) / name).chmod(0o600)
