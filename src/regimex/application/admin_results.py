"""Validate child artifacts and commit them under the original administrative attempt."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import JsonValue

from regimex.admin_contracts import (
    AdminRunRecord,
    AnnotationObject,
    CatalogBinding,
    DatasetRecord,
    EvaluationRecord,
    EvaluationSpec,
    GenerationSpec,
    ReferenceGenerationSpec,
    ReferenceSetRecord,
    TrainingSpec,
)
from regimex.application.service import AnalysisService
from regimex.contracts import StrictModel
from regimex.data_catalog import (
    SourceSnapshot,
    content_hash,
    create_snapshot_member,
    validate_reference_membership,
)
from regimex.directions import DirectionAnnotations
from regimex.exports import ExportV2
from regimex.infrastructure.object_store import ObjectInfo
from regimex.model_bundles import verify_model_bundle
from regimex.service_contracts import ModelRecord, Principal, SnapshotRecord
from regimex.storage import digest, json_bytes

if TYPE_CHECKING:
    from regimex.infrastructure.admin_repository import AdminRepository


def _model[T: StrictModel](value: object, model: type[T]) -> T:
    return model.model_validate_json(json_bytes(value))


def _dict(value: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError("expected a computation result object")
    return value


def _list(value: JsonValue) -> list[JsonValue]:
    if not isinstance(value, list):
        raise ValueError("expected a computation result list")
    return value


def _result_path(root: Path, value: JsonValue) -> Path:
    if (
        not isinstance(value, str)
        or Path(value).is_absolute()
        or ".." in Path(value).parts
        or not value.startswith("output/")
    ):
        raise ValueError("computation output path must remain under its private output directory")
    path = root / value
    if path.is_symlink() or not path.is_file():
        raise ValueError("computation output must be a regular file")
    return path


@dataclass(frozen=True, slots=True)
class AdminResultCommitter:
    """Keep verified object uploads separate from the transactional visibility boundary.

    Each instance belongs to one claimed attempt. Repository completion methods
    recheck its fence and resource identities in the same transaction as the commit.
    """

    analysis_service: AnalysisService
    repository: AdminRepository
    run: AdminRunRecord
    principal: Principal
    root: Path
    heartbeat: Callable[[], None]

    @property
    def _token(self) -> UUID:
        token = self.run.operation.attempt_token
        assert token is not None
        return token

    @property
    def _prefix(self) -> str:
        return f"admin/{self.run.run_id}/{self._token}"

    def _upload(self, value: JsonValue, key: str) -> ObjectInfo:
        path = _result_path(self.root, value)
        return self.analysis_service.store.put(key, path, expected_sha256=digest(path))

    def commit(self, result: dict[str, JsonValue]) -> AdminRunRecord:
        spec = self.run.spec
        if isinstance(spec, GenerationSpec):
            return self._generation(spec, result)
        if isinstance(spec, ReferenceGenerationSpec):
            return self._references(spec, result)
        if isinstance(spec, TrainingSpec):
            return self._training(spec, result)
        if isinstance(spec, EvaluationSpec):
            return self._evaluation(spec, result)
        if result != {"verified": True}:
            raise ValueError("publication requires a successful saved-model verification")
        self.heartbeat()
        return self.repository.complete_publication(self.run.run_id, self._token)

    def _generation(self, spec: GenerationSpec, result: dict[str, JsonValue]) -> AdminRunRecord:
        from regimex.data_catalog import DatasetManifest

        manifest = _model(result["manifest"], DatasetManifest)
        entries = _list(result["snapshots"])
        if manifest.dataset_id != spec.dataset_id or len(entries) != spec.plan.experiment_count:
            raise ValueError("generation output identity or experiment count differs")
        snapshots = []
        bindings = []
        for index, raw in enumerate(entries):
            item = _dict(raw)
            binding = _model(item["binding"], CatalogBinding)
            path = _result_path(self.root, item["path"])
            export = ExportV2.model_validate_json(path.read_bytes())
            member = manifest.members[index]
            if (
                create_snapshot_member(
                    SourceSnapshot(
                        source_namespace=binding.source_namespace,
                        split_group=binding.split_group,
                        export=export,
                    ),
                    member.partition,
                )
                != member
            ):
                raise ValueError("generated snapshot differs from sealed dataset membership")
            info = self._upload(item["path"], f"{self._prefix}/snapshot-{binding.snapshot_id}.json")
            snapshots.append(
                SnapshotRecord(
                    snapshot_id=binding.snapshot_id,
                    owner_id=self.principal.owner_id,
                    source_id=export.source.source_id,
                    run_id=export.source.run_id,
                    source_snapshot_id=export.source.snapshot_id,
                    content_sha256=export.content_sha256,
                    object=info,
                    created_at=datetime.now(UTC),
                )
            )
            bindings.append(binding)
            self.heartbeat()
        provenance = self._upload(
            result["provenance_path"], f"{self._prefix}/generation-provenance.json"
        )
        dataset = DatasetRecord(
            dataset_id=spec.dataset_id,
            owner_id=self.principal.owner_id,
            manifest=manifest,
            snapshot_ids=tuple(item.snapshot_id for item in snapshots),
            generation=spec,
            generation_provenance=provenance,
            created_at=datetime.now(UTC),
        )
        self.heartbeat()
        return self.repository.complete_generation(
            self.run.run_id,
            self._token,
            dataset=dataset,
            snapshots=tuple(snapshots),
            bindings=tuple(bindings),
        )

    def _references(
        self, spec: ReferenceGenerationSpec, result: dict[str, JsonValue]
    ) -> AdminRunRecord:
        from regimex.data_catalog import ReferenceSet

        references = _model(result["references"], ReferenceSet)
        dataset = self.repository.get_dataset(self.principal, spec.dataset_id)
        validate_reference_membership(dataset.manifest, references)
        if (
            references.reference_set_id != spec.reference_set_id
            or references.partition != spec.partition
            or references.analysis_profile != spec.analysis
            or references.kind != "automatic"
        ):
            raise ValueError("generated references differ from the immutable request")
        documents = _list(result["annotations"])
        if len(documents) != len(references.members):
            raise ValueError("generated references do not cover their complete partition")
        objects = []
        for index, (raw, reference_member) in enumerate(
            zip(documents, references.members, strict=True)
        ):
            item = _dict(raw)
            annotations = DirectionAnnotations.model_validate_json(
                _result_path(self.root, item["path"]).read_bytes()
            )
            if (
                content_hash(annotations.model_dump(mode="json"))
                != reference_member.annotations_sha256
            ):
                raise ValueError("generated annotation differs from its sealed reference identity")
            info = self._upload(item["path"], f"{self._prefix}/annotation-{index}.json")
            objects.append(
                AnnotationObject(
                    source_namespace=reference_member.source_namespace,
                    snapshot_id=reference_member.snapshot_id,
                    object=info,
                )
            )
            self.heartbeat()
        record = ReferenceSetRecord(
            dataset_id=dataset.dataset_id,
            owner_id=self.principal.owner_id,
            references=references,
            annotations=tuple(objects),
            created_at=datetime.now(UTC),
        )
        return self.repository.complete_reference(self.run.run_id, self._token, record)

    def _training(self, spec: TrainingSpec, result: dict[str, JsonValue]) -> AdminRunRecord:
        from regimex.application.training import CandidateTrainingResult

        candidate = _model(result["candidate"], CandidateTrainingResult)
        bundle_root = self.root / "output/bundle"
        descriptor = self.analysis_service.registry.get(spec.architecture_id)
        bundle_manifest = verify_model_bundle(
            bundle_root,
            manifest_sha256=candidate.manifest_sha256,
            architecture_id=spec.architecture_id,
            implementation_version=descriptor.implementation_version,
        )
        dataset = self.repository.get_dataset(self.principal, spec.dataset_id)
        if (
            bundle_manifest.model_version_id != spec.model_version_id
            or bundle_manifest.training.dataset_sha256 != dataset.manifest.content_sha256
            or bundle_manifest.training.run_id != str(self.run.run_id)
        ):
            raise ValueError("candidate bundle provenance differs from the training operation")
        manifest_object = self.analysis_service.store.put(
            f"{self._prefix}/bundle/manifest.json",
            bundle_root / "manifest.json",
            expected_sha256=candidate.manifest_sha256,
        )
        members = {}
        for bundle_member in bundle_manifest.members:
            members[bundle_member.path] = self.analysis_service.store.put(
                f"{self._prefix}/bundle/{bundle_member.path}",
                bundle_root / bundle_member.path,
                expected_sha256=bundle_member.sha256,
            )
            self.heartbeat()
        model = ModelRecord(
            model_version_id=spec.model_version_id,
            architecture_id=spec.architecture_id,
            bundle_sha256=candidate.manifest_sha256,
            manifest=manifest_object,
            members=members,
            inference_runtime=spec.inference_runtime,
            inference_policy_sha256=content_hash(spec.inference_runtime.model_dump(mode="json")),
            created_at=datetime.now(UTC),
        )
        return self.repository.complete_training(self.run.run_id, self._token, model)

    def _evaluation(self, spec: EvaluationSpec, result: dict[str, JsonValue]) -> AdminRunRecord:
        model = self.analysis_service.repository.get_model(spec.model_version_id)
        dataset = self.repository.get_dataset(self.principal, spec.dataset_id)
        evaluation_refs = self.repository.get_reference_set(
            self.principal, spec.reference_set_id, allow_test=spec.partition == "test"
        )
        metrics = _dict(result["metrics"])
        info = self._upload(result["report_path"], f"{self._prefix}/evaluation.json")
        if digest(_result_path(self.root, result["report_path"])) != content_hash(metrics):
            raise ValueError("evaluation report differs from its summary")
        evaluation = EvaluationRecord(
            run_id=self.run.run_id,
            owner_id=self.principal.owner_id,
            model_version_id=model.model_version_id,
            bundle_sha256=model.bundle_sha256,
            inference_policy_sha256=model.inference_policy_sha256,
            dataset_id=dataset.dataset_id,
            dataset_sha256=dataset.manifest.content_sha256,
            reference_set_id=evaluation_refs.references.reference_set_id,
            reference_sha256=evaluation_refs.references.content_sha256,
            partition=spec.partition,
            report=info,
            metrics=metrics,
            created_at=datetime.now(UTC),
        )
        self.heartbeat()
        return self.repository.complete_evaluation(self.run.run_id, self._token, evaluation)
