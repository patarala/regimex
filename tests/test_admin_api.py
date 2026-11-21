"""Administrative boundary checks and an actual isolated training lifecycle."""

import importlib.util
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
from uuid import UUID, uuid4

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx2")
psycopg = pytest.importorskip("psycopg")
pytest.importorskip("alembic")
pytest.importorskip("procrastinate")

from fastapi.testclient import TestClient
from httpx2 import Response
from pydantic import SecretStr

from regimex.admin_api import (
    AdminAPIService,
    AdminRunView,
    AutomaticReferenceRequest,
    TrainingRunRequest,
)
from regimex.admin_contracts import ReferenceGenerationSpec, RequestFingerprint
from regimex.api import AnalysisAPIService, create_app
from regimex.application.admin_service import AdminService
from regimex.application.benchmark_service import BenchmarkService
from regimex.application.service import AnalysisService
from regimex.benchmark_api import BenchmarkAPIService
from regimex.data_catalog import content_hash
from regimex.directions import AnalysisConfig
from regimex.infrastructure.admin_repository import AdminRepository
from regimex.infrastructure.database import DatabaseRepository
from regimex.infrastructure.s3_object_store import S3ObjectStore
from regimex.model_registry import ArchitectureRegistry
from regimex.service_bootstrap import create_object_store
from regimex.service_contracts import Principal
from regimex.service_settings import ServiceSettings
from regimex.storage import json_bytes


def test_reference_retries_preserve_published_hashes_and_explicit_profile_fields() -> None:
    body = {"mode": "automatic", "dataset_id": "DATASET_ID", "partition": "train"}
    current = "117093ab9a5e7a8aebdca2158c21666c0f23576a32447c93eb40130eb98f7cee"
    previous_21 = "3c0ea083342ef7253144818651cba2aeeea838828b5867c09a4c945a5298fdef"
    previous = "ce2c48baaaaa4d420bf5c12c0d0787e719a2ce0bba97f8cd79b60faaae19f4ed"

    def fingerprint(**fields: object) -> RequestFingerprint:
        return RequestFingerprint.for_profile_request(
            AutomaticReferenceRequest.model_validate({**body, **fields})
        )

    assert fingerprint() == RequestFingerprint(
        sha256=current, previous_hashes=(previous_21, previous)
    )
    assert fingerprint(analysis={"min_duration": 32}) == fingerprint()
    for explicit in (
        fingerprint(analysis={"version": "2.2"}),
        fingerprint(analysis_profile_id="direction-32/2.2"),
    ):
        assert explicit.sha256 == current
        assert (
            previous not in explicit.previous_hashes and previous_21 not in explicit.previous_hashes
        )
    assert (
        fingerprint(analysis={"version": "2.1"}, analysis_profile_id="direction-32/2.1").sha256
        == previous_21
    )
    changed = fingerprint(analysis={"min_duration": 64})
    assert previous not in (changed.sha256, *changed.previous_hashes)


def test_training_replay_fills_only_an_omitted_profile() -> None:
    body = {
        "dataset_id": "dataset",
        "labels_reference_set_id": "train",
        "validation_reference_set_id": "valid",
        "architecture_id": "baseline",
        "parameters": {"regularization": [1.0]},
        "policy": {"optimizer": "lbfgs", "seeds": [11]},
        "benchmark_selection_id": "bench",
    }
    request = TrainingRunRequest.model_validate_json(json_bytes(body))
    original = request.model_dump(mode="json")
    assert "display_name" not in original
    unnamed = TrainingRunRequest.model_validate_json(json_bytes({**body, "display_name": None}))
    assert RequestFingerprint.for_profile_request(
        unnamed
    ) == RequestFingerprint.for_profile_request(request)
    named = TrainingRunRequest.model_validate_json(json_bytes({**body, "display_name": "Trial 1"}))
    assert RequestFingerprint.for_profile_request(named).sha256 == content_hash(
        {**original, "display_name": "Trial 1"}
    )
    previous = tuple(
        content_hash({**original, "analysis_profile_id": f"direction-32/{version}"})
        for version in ("2.1", "2.0")
    )
    assert RequestFingerprint.for_profile_request(request) == RequestFingerprint(
        sha256=content_hash(original), previous_hashes=previous
    )
    changed = TrainingRunRequest.model_validate_json(json_bytes({**body, "parameters": {}}))
    fingerprint = RequestFingerprint.for_profile_request(changed)
    assert not set(previous).intersection((fingerprint.sha256, *fingerprint.previous_hashes))


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("POST", "/api/v1/models/{identifier}/publish", {"evaluation_run_id": str(uuid4())}),
        ("GET", "/api/v1/datasets/{identifier}", None),
        ("GET", "/api/v1/training-options/benchmarks?dataset_id={identifier}", None),
        ("GET", "/api/v1/training-options/benchmarks?dataset_id=desk%00north", None),
        ("GET", "/api/v1/reference-sets/{identifier}/annotations", None),
        ("POST", "/api/v1/benchmarks/{identifier}/open", None),
        ("GET", "/api/v1/benchmarks?benchmark_id=desk%00north", None),
        (
            "POST",
            "/api/v1/benchmarks",
            {"benchmark_id": "desk\x00north", "dataset_id": "dataset", "reference_set_id": "set"},
        ),
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


@pytest.mark.parametrize("version", ["2.0", "2.1"])
def test_reference_request_replays_the_original_profile_and_operation_after_upgrade(
    administrative_api: AdministrativeAPI, version: Literal["2.0", "2.1"]
) -> None:
    from test_admin_repository import dataset_records

    context = administrative_api
    repository = context.service.repository
    owner = context.analysis.authenticate(context.token)
    dataset, snapshots, bindings = dataset_records("dataset", owner=owner.owner_id)
    for snapshot in snapshots:
        repository.database.add_snapshot(snapshot)
    repository.add_dataset(owner, dataset, bindings=bindings)
    body = {"mode": "automatic", "dataset_id": "dataset", "partition": "train"}
    resolved = AutomaticReferenceRequest.model_validate(
        {**body, "analysis_profile_id": f"direction-32/{version}", "analysis": {"version": version}}
    )
    old_digest = content_hash(resolved.model_dump(mode="json"))
    spec = ReferenceGenerationSpec(
        dataset_id="dataset",
        partition="train",
        reference_set_id="first",
        analysis=AnalysisConfig(version=version),
        analysis_profile_id=resolved.analysis_profile_id,
    )
    original = repository.submit(owner, spec, idempotency_key="same", request_sha256=old_digest)
    headers = {**context.headers, "Idempotency-Key": "same"}
    response = context.client.post("/api/v1/reference-sets", headers=headers, json=body)
    assert response.status_code == 202, response.text
    assert response.json()["run_id"] == str(original.run_id)
    saved = repository.get_run_internal(original.run_id)
    assert saved == original and saved.operation.request_sha256 == old_digest
    # The transactional check must also replay when another submission wins
    # the race after the application's initial lookup.
    assert (
        repository.submit(
            owner,
            spec.model_copy(update={"reference_set_id": "concurrent"}),
            idempotency_key="same",
            request_sha256=RequestFingerprint.for_profile_request(
                AutomaticReferenceRequest.model_validate(body)
            ),
        )
        == original
    )
    changed = {
        **body,
        "analysis_profile_id": "direction-32/" + ("2.1" if version == "2.0" else "2.0"),
    }
    assert (
        context.client.post("/api/v1/reference-sets", headers=headers, json=changed).status_code
        == 409
    )


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


def test_cli_seed_groups_survive_snapshot_upload_and_dataset_import(
    administrative_api: AdministrativeAPI, tmp_path: Path
) -> None:
    pytest.importorskip("numpy")
    from typer.testing import CliRunner

    from regimex.cli import app
    from regimex.market_identity import market_seed_group

    context = administrative_api
    snapshots = []
    for seed, steps in ((17, 64), (17, 96), (22, 64), (31, 64)):
        path = tmp_path / f"{seed}-{steps}.json"
        args = ["generate", "--seed", str(seed), "--steps", str(steps), "--instruments", "1"]
        result = CliRunner().invoke(app, [*args, "--output", str(path)])
        assert result.exit_code == 0, result.output
        uploaded = context.client.post(
            "/api/v1/snapshots",
            content=path.read_bytes(),
            headers={**context.headers, "Content-Type": "application/json"},
        )
        assert uploaded.status_code == 201, uploaded.text
        snapshots.append(uploaded.json()["snapshot_id"])
    assert len(set(snapshots)) == 4

    def create(indices: tuple[int, int, int], *, override: bool = False) -> Response:
        members = [
            {"snapshot_id": snapshots[i], "partition": partition}
            for i, partition in zip(indices, ("train", "validation", "test"), strict=True)
        ]
        if override:
            members[1]["split_group"] = market_seed_group(123).model_dump(mode="json")
        return context.client.post(
            "/api/v1/datasets",
            json={"mode": "snapshots", "members": members},
            headers=context.headers,
        )

    assert create((0, 1, 2)).status_code == 422
    assert create((0, 1, 2), override=True).status_code == 422
    distinct = create((0, 2, 3))
    assert distinct.status_code == 201, distinct.text
    manifest = context.client.get(
        f"/api/v1/datasets/{distinct.json()['dataset_id']}/manifest", headers=context.headers
    )
    assert manifest.status_code == 200, manifest.text
    assert [member["split_group"] for member in manifest.json()["members"]] == [
        market_seed_group(seed).model_dump(mode="json") for seed in (17, 22, 31)
    ]


def test_previously_bound_sample_keeps_its_historical_group(
    administrative_api: AdministrativeAPI,
) -> None:
    pytest.importorskip("numpy")
    from uuid import NAMESPACE_URL, uuid5

    from regimex.admin_contracts import CatalogBinding, DatasetSelection, DatasetSpec
    from regimex.data_catalog import SplitGroup
    from regimex.exports import ExportPayloadV2, seal_export
    from regimex.market_export import sample_snapshot
    from regimex.market_identity import market_seed_group
    from regimex.service_contracts import Conflict

    context = administrative_api
    owner = context.analysis.authenticate(context.token)
    export = sample_snapshot(seed=17, steps=64, instruments=1)
    value = export.model_dump(mode="json", exclude={"content_sha256"})
    run_id = str(uuid5(NAMESPACE_URL, f"regimex:single-market:{export.source.configuration_id}"))
    value["source"].update(run_id=run_id, snapshot_id=run_id)
    old = seal_export(ExportPayloadV2.model_validate_json(json_bytes(value)))
    uploaded = context.client.post(
        "/api/v1/snapshots",
        content=old.model_dump_json(),
        headers={**context.headers, "Content-Type": "application/json"},
    )
    assert uploaded.status_code == 201, uploaded.text
    snapshot_id = UUID(uploaded.json()["snapshot_id"])
    binding = CatalogBinding(
        snapshot_id=snapshot_id,
        source_namespace=content_hash([owner.owner_id, old.source.source_id, run_id]),
        split_group=SplitGroup(
            namespace=content_hash([owner.owner_id, old.source.source_id]), group_id=run_id
        ),
    )
    context.service.repository.add_binding(owner, binding)
    spec = DatasetSpec(
        dataset_id="new-dataset",
        members=(
            DatasetSelection(
                snapshot_id=snapshot_id, partition="train", split_group=market_seed_group(17)
            ),
            DatasetSelection(snapshot_id=uuid4(), partition="validation"),
            DatasetSelection(snapshot_id=uuid4(), partition="test"),
        ),
    )
    with pytest.raises(Conflict, match="registered group"):
        context.service.create_dataset(owner, spec)
    assert context.service.repository.get_binding(owner, snapshot_id) == binding
    assert context.analysis.read_snapshot(owner, snapshot_id) == old


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
    listed = context.client.get(
        "/api/v1/reference-sets", params={"dataset_id": dataset_id}, headers=headers
    )
    assert listed.status_code == 200
    rows = listed.json()["items"]
    assert {row["reference_set_id"] for row in rows} == {
        references["train"],
        references["validation"],
    }
    assert all(row["current_training_profile"] for row in rows)
    assert (
        context.client.get(
            "/api/v1/reference-sets", params={"dataset_id": final_dataset_id}, headers=headers
        ).json()["items"]
        == []
    )
    history = context.client.post(
        "/api/v1/benchmarks/history/datasets", headers=headers, json={"dataset_id": dataset_id}
    )
    assert history.status_code == 200, history.text
    registered = context.client.get(
        "/api/v1/datasets", params={"development_only": True}, headers=headers
    )
    assert registered.status_code == 200
    assert [item["dataset_id"] for item in registered.json()["items"]] == [dataset_id]
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
    benchmark_id = "東京/November + models" if architecture == "tcn" else "registry"
    benchmark_query = {"benchmark_id": benchmark_id}
    reservation = context.client.post(
        "/api/v1/benchmarks",
        headers=headers,
        json={
            "benchmark_id": benchmark_id,
            "dataset_id": final_dataset_id,
            "reference_set_id": final_reference_id,
        },
    )
    assert reservation.status_code == 201, reservation.text
    detail = context.client.get(reservation.headers["Location"], headers=headers)
    assert detail.status_code == 200 and detail.json() == reservation.json()

    def assert_benchmark_options(expected_ids: list[str]) -> None:
        options = context.client.get(
            "/api/v1/training-options/benchmarks",
            params={"dataset_id": dataset_id},
            headers=headers,
        )
        assert options.status_code == 200, options.text
        assert options.json() == {"benchmark_ids": expected_ids}

    assert_benchmark_options([])
    started = context.client.post(
        "/api/v1/benchmarks/selection-start", params=benchmark_query, headers=headers
    )
    assert started.status_code == 200, started.text
    assert_benchmark_options([benchmark_id])
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
        "benchmark_selection_id": benchmark_id,
    }
    training_headers = context.headers
    if tcn:
        training["display_name"] = "  Short windows  "
    response = context.client.post("/api/v1/training-runs", json=training, headers=training_headers)
    assert response.status_code == 202, response.text
    run = AdminRunView.model_validate_json(response.content)
    assert run.training is not None and run.training.dataset_id == dataset_id
    assert (run.display_name, run.name_revision) == ("Short windows" if tcn else None, 0)
    run_path = f"/api/v1/training-runs/{run.run_id}"
    progress = context.client.get(run_path + "/progress", headers=headers)
    assert progress.status_code == 200, progress.text
    assert progress.json() == {"attempt": 0, "cursor": 0, "has_more": False, "history": {}}
    assert context.client.get(run_path + "/metrics", headers=headers).status_code == 409
    completed = context.service.run_admin(run.run_id)
    assert completed is not None
    assert completed.operation.state == "succeeded", completed.operation.model_dump_json()
    assert completed.output is not None
    renamed = context.client.patch(
        run_path + "/name",
        json={"display_name": "Published candidate", "expected_revision": 0},
        headers=headers,
    )
    assert renamed.status_code == 200, renamed.text
    assert renamed.json() == {
        "run_id": str(run.run_id),
        "display_name": "Published candidate",
        "revision": 1,
    }
    detail = context.client.get(run_path, headers=headers).json()
    assert detail["display_name"] == "Published candidate" and detail["name_revision"] == 1
    assert detail["training"] == run.training.model_dump(mode="json")
    model_id = completed.output["model_version_id"]
    assert isinstance(model_id, str)
    metrics = context.client.get(run_path + "/metrics", headers=headers)
    assert metrics.status_code == 200, metrics.text
    assert metrics.json()["model_version_id"] == model_id
    assert 0 <= metrics.json()["validation_segment_macro_f1"] <= 1
    assert metrics.json()["history"]["epochs" if tcn else "candidates"]
    user_headers = {"Authorization": f"Bearer {context.user_token}"}
    for part in ("metrics", "progress", "actions"):
        assert context.client.get(run_path + "/" + part, headers=user_headers).status_code == 403
    assert (
        context.client.get(
            f"/api/v1/models/{model_id}", headers={"Authorization": f"Bearer {context.user_token}"}
        ).status_code
        == 404
    )
    validation_request = {
        "model_version_id": model_id,
        "dataset_id": dataset_id,
        "reference_set_id": references["validation"],
    }
    response = context.client.post(
        "/api/v1/evaluations", json=validation_request, headers=context.headers
    )
    assert response.status_code == 202, response.text
    evaluation = AdminRunView.model_validate_json(response.content)
    completed_evaluation = context.service.run_admin(evaluation.run_id)
    assert (
        completed_evaluation is not None and completed_evaluation.operation.state == "succeeded"
    ), completed_evaluation
    again_headers = context.headers
    repeated = context.client.post(
        "/api/v1/evaluations", json=validation_request, headers=again_headers
    )
    assert (
        repeated.status_code == 202 and repeated.headers["location"] == response.headers["location"]
    )
    assert repeated.json()["job"]["state"] == "succeeded"
    conflicting = context.client.post(
        "/api/v1/evaluations",
        json={**validation_request, "reference_set_id": "another-set"},
        headers=again_headers,
    )
    assert conflicting.status_code == 409
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
        "/api/v1/benchmarks/freeze",
        params=benchmark_query,
        headers=headers,
        json={"model_version_id": model_id},
    )
    assert frozen.status_code == 200, frozen.text
    assert_benchmark_options([])
    assert (
        context.client.post(
            "/api/v1/training-runs", json=training, headers=context.headers
        ).status_code
        == 403
    )
    replay = context.client.post("/api/v1/training-runs", json=training, headers=training_headers)
    assert replay.status_code == 202 and replay.json()["run_id"] == str(run.run_id)
    assert replay.json()["display_name"] == "Published candidate"
    unopened = context.client.post(
        "/api/v1/benchmarks/evaluations",
        params=benchmark_query,
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
    opened = context.client.post("/api/v1/benchmarks/open", params=benchmark_query, headers=headers)
    assert opened.status_code == 200, opened.text
    final_evaluation_response = context.client.post(
        "/api/v1/benchmarks/evaluations",
        params=benchmark_query,
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
    rows = context.client.get(run_path + "/actions", headers=headers).json()["items"]
    assert [row["job"]["kind"] for row in rows] == ["publication", "evaluation"]
    assert rows[1]["run_id"] == str(evaluation.run_id)
