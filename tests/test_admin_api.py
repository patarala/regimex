"""Administrative boundary checks and an actual isolated training lifecycle."""

import importlib.util
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx2")
psycopg = pytest.importorskip("psycopg")
pytest.importorskip("alembic")
pytest.importorskip("procrastinate")

from fastapi.testclient import TestClient
from pydantic import SecretStr

from regimex.admin_api import AdminAPIService, AdminRunView
from regimex.api import AnalysisAPIService, create_app
from regimex.application.admin_service import AdminService
from regimex.application.benchmark_service import BenchmarkService
from regimex.application.service import AnalysisService
from regimex.benchmark_api import BenchmarkAPIService
from regimex.infrastructure.admin_repository import AdminRepository
from regimex.infrastructure.database import DatabaseRepository
from regimex.infrastructure.s3_object_store import S3ObjectStore
from regimex.model_registry import ArchitectureRegistry
from regimex.service_bootstrap import create_object_store
from regimex.service_contracts import Principal
from regimex.service_settings import ServiceSettings


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("POST", "/api/v1/models/{identifier}/publish", {"evaluation_run_id": str(uuid4())}),
        ("GET", "/api/v1/datasets/{identifier}", None),
        ("GET", "/api/v1/reference-sets/{identifier}/annotations", None),
        ("POST", "/api/v1/benchmarks/{identifier}/open", None),
        ("GET", "/api/v1/metrics?model_version_id={identifier}", None),
    ],
)
def test_invalid_resource_identifiers_fail_at_http_boundary_before_application_calls(
    method: str, path: str, body: dict[str, str] | None
) -> None:
    class AuthenticationOnly:
        def authenticate(self, _raw_token: str) -> Principal:
            return Principal(owner_id="admin", role="admin", token_id=uuid4())

    # Application methods are deliberately absent: malformed identifiers must
    # never reach persistence or raise an internal DTO validation exception.
    application = create_app(
        cast(AnalysisAPIService, AuthenticationOnly()),
        admin_service=cast(AdminAPIService, object()),
        benchmark_service=cast(BenchmarkAPIService, object()),
    )
    with TestClient(application) as client:
        response = client.request(
            method,
            path.format(identifier="x" * 129),
            json=body,
            headers={"Authorization": "Bearer fixture", "Idempotency-Key": "invalid-identifier"},
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_data"
    assert "x" * 129 not in response.text


def test_complete_api_factory_and_schemas_do_not_import_scientific_libraries(
    tmp_path: Path,
) -> None:
    code = """
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'numpy','sklearn','torch','matplotlib','plotly'}:
            raise AssertionError('optional import at API startup: '+fullname)
sys.meta_path.insert(0,Block())
from regimex.service_bootstrap import create_app
app=create_app()
schema=app.openapi()
assert '/api/v1/training-runs' in schema['paths']
assert '/api/v1/models/{model_version_id}/publish' in schema['paths']
def check(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if key=='$ref' and item.startswith('#/components/schemas/'):
                assert item.rsplit('/',1)[-1] in schema['components']['schemas'],item
            check(item)
    elif isinstance(value,list):
        for item in value: check(item)
check(schema)
assert not {'numpy','sklearn','torch','matplotlib','plotly'} & sys.modules.keys()
"""
    subprocess.run(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "REGIMEX_DATABASE_DSN": "dbname=not_connected",
            "REGIMEX_OBJECT_ROOT": str(tmp_path / "objects"),
            "REGIMEX_WORK_ROOT": str(tmp_path / "work"),
        },
        check=True,
        capture_output=True,
        text=True,
    )


def test_s3_api_factory_needs_no_sdk_credentials_or_provider_contact(tmp_path: Path) -> None:
    code = """
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'boto3','botocore','numpy','sklearn','torch'}:
            raise AssertionError('eager dependency or provider contact: '+fullname)
sys.meta_path.insert(0,Block())
from regimex.service_bootstrap import create_app
app=create_app()
assert '/api/v1/analyses' in app.openapi()['paths']
assert not {'boto3','botocore','numpy','sklearn','torch'} & sys.modules.keys()
"""
    subprocess.run(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "REGIMEX_DATABASE_DSN": "dbname=not_connected",
            "REGIMEX_OBJECT_BACKEND": "s3",
            "REGIMEX_S3_ENDPOINT_URL": "https://objects.example.invalid",
            "REGIMEX_S3_REGION": "us-east-1",
            "REGIMEX_S3_BUCKET": "private-fixtures",
            "REGIMEX_S3_PREFIX": "offline-startup",
            "REGIMEX_OBJECT_ROOT": str(tmp_path / "unused-objects"),
            "REGIMEX_WORK_ROOT": str(tmp_path / "work"),
        },
        check=True,
        capture_output=True,
        text=True,
    )


@dataclass
class AdministrativeAPI:
    client: TestClient
    service: AdminService
    analysis: AnalysisService
    benchmark: BenchmarkService
    token: str
    user_token: str

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}", "Idempotency-Key": uuid4().hex}


@pytest.fixture(params=("local", "s3"))
def administrative_api(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Iterator[AdministrativeAPI]:
    from test_object_store_contract import cleanup_owned_test_prefix, s3_configuration

    dsn = os.environ.get("REGIMEX_TEST_DATABASE_DSN")
    if not dsn:
        pytest.skip("set REGIMEX_TEST_DATABASE_DSN to a disposable PostgreSQL database")
    config = s3_configuration(tmp_path / "coordination") if request.param == "s3" else None
    schema = "regimex_admin_api_" + uuid4().hex
    database = DatabaseRepository(dsn, schema=schema)
    database.migrate()
    settings = ServiceSettings(
        database_dsn=SecretStr(dsn),
        database_schema=schema,
        object_root=tmp_path / "objects",
        object_backend="s3" if config is not None else "local",
        s3=config,
        work_root=tmp_path / "work",
        max_object_bytes=16 * 1024**2,
        max_store_bytes=64 * 1024**2,
        admin_time_budget_seconds=90,
    )
    objects = create_object_store(settings)
    analysis = AnalysisService(
        database,
        objects,
        ArchitectureRegistry.discover(),
        tmp_path / "work",
        heartbeat_seconds=0.05,
    )
    administration = AdminRepository(database)
    benchmark = BenchmarkService(database, administration, analysis.store, analysis.work_root)
    admin = AdminService(
        analysis,
        administration,
        settings,
        authorize_selection=benchmark.authorize_selection,
        authorize_final_evaluation=benchmark.authorize_final_evaluation,
    )
    token = database.create_token("fixture-administrator", "admin")
    user = database.create_token("fixture-user", "user")
    try:
        with TestClient(
            create_app(analysis, admin_service=admin, benchmark_service=benchmark)
        ) as client:
            yield AdministrativeAPI(client, admin, analysis, benchmark, token, user)
    finally:
        for root, directories, files in os.walk(tmp_path / "work"):
            Path(root).chmod(0o700)
            for name in directories:
                (Path(root) / name).chmod(0o700)
            for name in files:
                (Path(root) / name).chmod(0o600)
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(
                psycopg.sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    psycopg.sql.Identifier(schema)
                )
            )
        if isinstance(objects, S3ObjectStore):
            cleanup_owned_test_prefix(objects)


def test_administrative_intake_is_strict_authenticated_and_bounded(
    administrative_api: AdministrativeAPI,
) -> None:
    context = administrative_api
    body = {"mode": "generate", "plan": {"seed": 991, "experiment_count": 3, "step_count": 64}}
    assert context.client.post("/api/v1/datasets", json=body).status_code == 401
    assert (
        context.client.post(
            "/api/v1/datasets",
            json=body,
            headers={
                "Authorization": f"Bearer {context.user_token}",
                "Idempotency-Key": "user-forbidden",
            },
        ).status_code
        == 403
    )
    assert (
        context.client.post(
            "/api/v1/datasets",
            json={**body, "dataset_id": "caller-owned-id"},
            headers=context.headers,
        ).status_code
        == 422
    )
    assert (
        context.client.post(
            "/api/v1/datasets",
            json={
                "mode": "generate",
                "plan": {"seed": 991, "experiment_count": 3, "step_count": 64, "labeling": {}},
            },
            headers=context.headers,
        ).status_code
        == 422
    )
    oversized = {
        "mode": "generate",
        "plan": {"seed": 992, "experiment_count": 129, "step_count": 64},
    }
    assert (
        context.client.post("/api/v1/datasets", json=oversized, headers=context.headers).status_code
        == 429
    )


def test_generation_queues_once_and_uses_server_identity(
    administrative_api: AdministrativeAPI,
) -> None:
    context = administrative_api
    body = {"mode": "generate", "plan": {"seed": 993, "experiment_count": 3, "step_count": 64}}
    headers = context.headers
    first = context.client.post("/api/v1/datasets", json=body, headers=headers)
    assert first.status_code == 202, first.text
    view = AdminRunView.model_validate_json(first.content)
    replay = context.client.post("/api/v1/datasets", json=body, headers=headers)
    assert replay.status_code == 202 and replay.json() == first.json()
    assert view.job.kind == "generation" and view.job.state == "queued"
    assert "attempt_token" not in first.text and "object_key" not in first.text
    polling = context.client.get(view.polling_url, headers=headers)
    assert polling.status_code == 200 and polling.json() == first.json()
    changed = {"mode": "generate", "plan": {"seed": 994, "experiment_count": 3, "step_count": 64}}
    assert context.client.post("/api/v1/datasets", json=changed, headers=headers).status_code == 409
    with context.analysis.repository.connection() as connection:
        assert connection.execute(
            "SELECT task_name,queue_name FROM procrastinate_jobs"
        ).fetchall() == [("regimex.admin", "admin")]


def test_snapshot_dataset_import_binds_the_owner_idempotency_key(
    administrative_api: AdministrativeAPI,
) -> None:
    from test_admin_repository import sources

    context = administrative_api
    snapshots = []
    for item in sources():
        uploaded = context.client.post(
            "/api/v1/snapshots",
            content=item.export.model_dump_json(),
            headers={**context.headers, "Content-Type": "application/json"},
        )
        assert uploaded.status_code == 201, uploaded.text
        snapshots.append(uploaded.json()["snapshot_id"])
    body = {
        "mode": "snapshots",
        "members": [
            {"snapshot_id": snapshot, "partition": partition}
            for snapshot, partition in zip(snapshots, ("train", "validation", "test"), strict=True)
        ],
    }
    headers = context.headers
    created = context.client.post("/api/v1/datasets", json=body, headers=headers)
    assert created.status_code == 201, created.text
    replay = context.client.post("/api/v1/datasets", json=body, headers=headers)
    assert replay.status_code == 201 and replay.json() == created.json()
    changed = {
        "mode": "snapshots",
        "members": [
            {"snapshot_id": snapshot, "partition": partition}
            for snapshot, partition in zip(snapshots, ("validation", "train", "test"), strict=True)
        ],
    }
    conflict = context.client.post("/api/v1/datasets", json=changed, headers=headers)
    assert conflict.status_code == 409, conflict.text
    distinct = context.client.post("/api/v1/datasets", json=changed, headers=context.headers)
    assert distinct.status_code == 201, distinct.text
    assert distinct.json()["dataset_id"] != created.json()["dataset_id"]


@pytest.mark.parametrize("architecture", ["baseline", "tcn"])
def test_http_lifecycle_trains_evaluates_publishes_and_analyzes(
    administrative_api: AdministrativeAPI,
    architecture: str,
) -> None:
    if importlib.util.find_spec("numpy") is None or importlib.util.find_spec("sklearn") is None:
        pytest.skip("install scientific dependencies")
    if architecture == "tcn" and importlib.util.find_spec("torch") is None:
        pytest.skip("install CPU Torch")
    from test_application_training import source

    context = administrative_api
    headers = context.headers
    snapshots = []
    for index in (0, 1, 2):
        response = context.client.post(
            "/api/v1/snapshots",
            content=source(index).export.model_dump_json(),
            headers={**headers, "Content-Type": "application/json"},
        )
        assert response.status_code == 201, response.text
        snapshots.append(response.json()["snapshot_id"])
    response = context.client.post(
        "/api/v1/datasets",
        json={
            "mode": "snapshots",
            "members": [
                {"snapshot_id": snapshot, "partition": partition}
                for snapshot, partition in zip(
                    snapshots, ("train", "validation", "test"), strict=True
                )
            ],
        },
        headers=context.headers,
    )
    assert response.status_code == 201, response.text
    dataset_id = response.json()["dataset_id"]
    references = {}
    for partition in ("train", "validation", "test"):
        response = context.client.post(
            "/api/v1/reference-sets",
            json={"mode": "automatic", "dataset_id": dataset_id, "partition": partition},
            headers=context.headers,
        )
        assert response.status_code == 202, response.text
        queued = AdminRunView.model_validate_json(response.content)
        result = context.service.run_admin(queued.run_id)
        assert result is not None
        assert result.operation.state == "succeeded", result.operation.model_dump_json()
        assert result.output is not None
        references[partition] = result.output["reference_set_id"]
    assert (
        context.client.get(
            f"/api/v1/reference-sets/{references['test']}/annotations", headers=headers
        ).status_code
        == 403
    )
    # A separate, predeclared artificial benchmark exercises the production gate.
    # Its outcomes never select fixture parameters, and establish no model quality.
    final_snapshots = []
    for index in (3, 4, 5):
        uploaded = context.client.post(
            "/api/v1/snapshots",
            content=source(index).export.model_dump_json(),
            headers={**headers, "Content-Type": "application/json"},
        )
        assert uploaded.status_code == 201, uploaded.text
        final_snapshots.append(uploaded.json()["snapshot_id"])
    final_dataset_response = context.client.post(
        "/api/v1/datasets",
        headers=context.headers,
        json={
            "mode": "snapshots",
            "members": [
                {"snapshot_id": snapshot_id, "partition": partition}
                for snapshot_id, partition in zip(
                    final_snapshots, ("train", "validation", "test"), strict=True
                )
            ],
        },
    )
    assert final_dataset_response.status_code == 201, final_dataset_response.text
    final_dataset_id = final_dataset_response.json()["dataset_id"]
    reference_response = context.client.post(
        "/api/v1/reference-sets",
        headers=context.headers,
        json={"mode": "automatic", "dataset_id": final_dataset_id, "partition": "test"},
    )
    assert reference_response.status_code == 202, reference_response.text
    reference_run = AdminRunView.model_validate_json(reference_response.content)
    final_reference = context.service.run_admin(reference_run.run_id)
    assert final_reference is not None
    assert final_reference.operation.state == "succeeded", (
        final_reference.operation.model_dump_json()
    )
    assert final_reference.output is not None
    final_reference_id = final_reference.output["reference_set_id"]
    history = context.client.post(
        "/api/v1/benchmarks/history/datasets", headers=headers, json={"dataset_id": dataset_id}
    )
    assert history.status_code == 200, history.text
    namespaces = sorted(
        {
            member["source_namespace"]
            for identity in (dataset_id, final_dataset_id)
            for member in context.client.get(
                f"/api/v1/datasets/{identity}/manifest", headers=headers
            ).json()["members"]
        }
    )
    attestation = context.client.post(
        "/api/v1/benchmarks/attest",
        headers=headers,
        json={
            "source_namespaces": namespaces,
            "statement": "Complete declared fixture inventory in this disposable schema.",
        },
    )
    assert attestation.status_code == 200, attestation.text
    reservation = context.client.post(
        "/api/v1/benchmarks",
        headers=headers,
        json={
            "benchmark_id": "software-control",
            "dataset_id": final_dataset_id,
            "reference_set_id": final_reference_id,
        },
    )
    assert reservation.status_code == 201, reservation.text
    started = context.client.post(
        "/api/v1/benchmarks/software-control/selection-start", headers=headers
    )
    assert started.status_code == 200, started.text
    tcn = architecture == "tcn"
    training = {
        "dataset_id": dataset_id,
        "labels_reference_set_id": references["train"],
        "validation_reference_set_id": references["validation"],
        "architecture_id": architecture,
        "parameters": {"channels": 4, "dilations": [1, 2], "dropout": 0.0, "smoothing": 0.0}
        if tcn
        else {"regularization": [1.0], "context_sizes": [4, 16]},
        "policy": {
            "optimizer": "adamw" if tcn else "lbfgs",
            "seeds": [11],
            **(
                {"epochs": 1, "batch_size": 1, "patience": 1, "validation_interval": 1}
                if tcn
                else {}
            ),
        },
        "training_runtime": {"threads": 1, "time_budget_seconds": 90},
        "inference_runtime": {"threads": 1, "max_batch_size": 1},
        "benchmark_selection_id": "software-control",
    }
    training_headers = context.headers
    response = context.client.post("/api/v1/training-runs", json=training, headers=training_headers)
    assert response.status_code == 202, response.text
    run = AdminRunView.model_validate_json(response.content)
    completed = context.service.run_admin(run.run_id)
    assert completed is not None
    assert completed.operation.state == "succeeded", completed.operation.model_dump_json()
    assert completed.output is not None
    model_id = completed.output["model_version_id"]
    assert isinstance(model_id, str)
    assert (
        context.client.get(
            f"/api/v1/models/{model_id}", headers={"Authorization": f"Bearer {context.user_token}"}
        ).status_code
        == 404
    )
    response = context.client.post(
        "/api/v1/evaluations",
        json={
            "model_version_id": model_id,
            "dataset_id": dataset_id,
            "reference_set_id": references["validation"],
        },
        headers=context.headers,
    )
    assert response.status_code == 202, response.text
    evaluation = AdminRunView.model_validate_json(response.content)
    completed_evaluation = context.service.run_admin(evaluation.run_id)
    assert (
        completed_evaluation is not None and completed_evaluation.operation.state == "succeeded"
    ), completed_evaluation
    response = context.client.post(
        f"/api/v1/models/{model_id}/publish",
        json={"evaluation_run_id": str(evaluation.run_id)},
        headers=context.headers,
    )
    assert response.status_code == 202, response.text
    publication = AdminRunView.model_validate_json(response.content)
    published = context.service.run_admin(publication.run_id)
    assert published is not None and published.operation.state == "succeeded", published
    response = context.client.post(
        "/api/v1/analyses",
        json={"snapshot_id": snapshots[1], "model_version_id": "active"},
        headers=context.headers,
    )
    assert response.status_code == 202, response.text
    analysis_id = UUID(response.json()["analysis_id"])
    analyzed = context.analysis.run_analysis(analysis_id)
    assert analyzed is not None and analyzed.state == "succeeded", analyzed
    prediction_response = context.client.get(
        f"/api/v1/analyses/{analysis_id}/result", headers=headers
    )
    assert prediction_response.status_code == 200
    assert prediction_response.json()["architecture_id"] == architecture
    assert len(prediction_response.json()["sequences"][0]["probabilities"]) == 192
    metrics = context.client.get(
        "/api/v1/metrics", headers={"Authorization": f"Bearer {context.user_token}"}
    )
    assert metrics.status_code == 200 and metrics.json()["quality_status"] == "validation_only"
    frozen = context.client.post(
        "/api/v1/benchmarks/software-control/freeze",
        headers=headers,
        json={"model_version_id": model_id},
    )
    assert frozen.status_code == 200, frozen.text
    assert (
        context.client.post(
            "/api/v1/training-runs", json=training, headers=context.headers
        ).status_code
        == 403
    )
    replay = context.client.post("/api/v1/training-runs", json=training, headers=training_headers)
    assert replay.status_code == 202 and replay.json()["run_id"] == str(run.run_id)
    unopened = context.client.post(
        "/api/v1/benchmarks/software-control/evaluations",
        headers=context.headers,
        json={"model_version_id": model_id},
    )
    assert unopened.status_code == 403, unopened.text
    assert (
        context.client.post(
            "/api/v1/evaluations",
            headers=context.headers,
            json={
                "model_version_id": model_id,
                "dataset_id": final_dataset_id,
                "reference_set_id": final_reference_id,
                "partition": "test",
            },
        ).status_code
        == 422
    )
    opened = context.client.post("/api/v1/benchmarks/software-control/open", headers=headers)
    assert opened.status_code == 200, opened.text
    final_evaluation_response = context.client.post(
        "/api/v1/benchmarks/software-control/evaluations",
        headers=context.headers,
        json={"model_version_id": model_id},
    )
    assert final_evaluation_response.status_code == 202, final_evaluation_response.text
    final_evaluation = AdminRunView.model_validate_json(final_evaluation_response.content)
    final_completed = context.service.run_admin(final_evaluation.run_id)
    assert final_completed is not None and final_completed.operation.state == "succeeded", (
        final_completed
    )
    final_report = context.client.get(
        f"/api/v1/evaluations/{final_evaluation.run_id}/result", headers=headers
    )
    assert final_report.status_code == 200, final_report.text
    assert final_report.json()["reference_kind"] == "automatic"
    assert final_report.json()["quality_status"] == "validation_only"
    assert final_report.json()["independent"] is False
