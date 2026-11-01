"""Authorized service workflows without scientific imports in the HTTP process."""

import hashlib
import math
import os
import stat
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal
from uuid import UUID, uuid4

from pydantic import Field, JsonValue, model_validator

from regimex.contracts import MAX_INPUT_BYTES, MAX_STEPS, Identifier, StrictModel
from regimex.directions import AnalysisConfig, DirectionInterval
from regimex.exports import (
    ExportInstrument,
    ExportV2,
    ModelTimeGrid,
    SourceIdentity,
    parse_export_v2,
)
from regimex.infrastructure.object_store import (
    ObjectCapacityError,
    ObjectInfo,
    ObjectIntegrityError,
    ObjectStore,
)
from regimex.model_bundles import MAX_BUNDLE_MANIFEST_BYTES, ModelBundle, verify_model_bundle
from regimex.model_registry import ArchitectureInfo, ArchitectureRegistry, InferenceRuntime
from regimex.operations import Operation, ResolvedModel
from regimex.report_contracts import MAX_REPORT_BYTES, AnalysisReportRecord
from regimex.service_contracts import (
    AnalysisRecord,
    CapacityExceeded,
    Conflict,
    Forbidden,
    ModelRecord,
    NotFound,
    Principal,
    SnapshotRecord,
    UnsupportedContract,
)
from regimex.storage import json_bytes

if TYPE_CHECKING:
    from regimex.infrastructure.database import DatabaseRepository
    from regimex.service_contracts import AnalysisSummary


class PredictionSequence(StrictModel):
    """Predictions preserve every input bin, including unavailable price periods."""

    instrument_id: Identifier
    probabilities: Annotated[
        tuple[tuple[float, float, float], ...], Field(min_length=1, max_length=MAX_STEPS)
    ]
    intervals: Annotated[tuple[DirectionInterval, ...], Field(min_length=1, max_length=MAX_STEPS)]

    @model_validator(mode="after")
    def full_grid(self) -> PredictionSequence:
        if not self.probabilities or not self.intervals:
            raise ValueError("predictions require the complete nonempty observation grid")
        for row in self.probabilities:
            if any(value < 0 or value > 1 for value in row) or abs(sum(row) - 1) > 1e-6:
                raise ValueError("probability rows must be normalized and bounded")
        previous = 0
        for interval in self.intervals:
            if interval.instrument_id != self.instrument_id or interval.start != previous:
                raise ValueError("prediction intervals must cover one instrument without gaps")
            previous = interval.stop
        if previous != len(self.probabilities):
            raise ValueError("prediction intervals must cover every probability row")
        return self


class AnalysisResult(StrictModel):
    result_version: Literal["2.0"] = "2.0"
    analysis_id: UUID
    snapshot_id: UUID
    snapshot_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    source: SourceIdentity
    model: ResolvedModel
    architecture_id: Identifier
    implementation_version: Identifier
    decoder_version: Identifier
    analysis_profile_id: Identifier
    analysis: AnalysisConfig
    inference_runtime: InferenceRuntime
    class_names: tuple[Literal["sideways"], Literal["growth"], Literal["decline"]]
    time_grid: ModelTimeGrid
    instruments: Annotated[tuple[ExportInstrument, ...], Field(min_length=1, max_length=2)]
    sequences: Annotated[tuple[PredictionSequence, ...], Field(min_length=1, max_length=2)]

    @model_validator(mode="after")
    def original_grid(self) -> AnalysisResult:
        if tuple(item.instrument_id for item in self.instruments) != tuple(
            item.instrument_id for item in self.sequences
        ) or any(len(item.probabilities) != self.time_grid.step_count for item in self.sequences):
            raise ValueError("results must preserve the complete input instrument/time grid")
        return self


def materialize_object(store: ObjectStore, info: ObjectInfo, destination: Path) -> None:
    """Verify committed generation as well as bytes before exposing stored input."""
    actual = store.materialize(
        info.key,
        destination,
        expected_sha256=info.sha256,
        expected_size_bytes=info.size_bytes,
    )
    if actual != info:
        raise ObjectIntegrityError("object generation differs from its committed metadata")


def materialize_bundle(
    store: ObjectStore,
    model: ModelRecord,
    destination: Path,
    *,
    before_copy: Callable[[ObjectInfo], None] | None = None,
) -> None:
    if model.manifest.size_bytes > MAX_BUNDLE_MANIFEST_BYTES:
        raise ObjectIntegrityError("stored bundle manifest exceeds its input size limit")
    destination.mkdir(mode=0o700)
    if before_copy is not None:
        before_copy(model.manifest)
    materialize_object(store, model.manifest, destination / "manifest.json")
    manifest = ModelBundle.model_validate_json((destination / "manifest.json").read_bytes())
    expected = {member.path: member for member in manifest.members}
    if set(expected) != set(model.members):
        raise ObjectIntegrityError("stored bundle membership differs from its manifest")
    for name, info in model.members.items():
        member = expected[name]
        if (info.sha256, info.size_bytes) != (member.sha256, member.size_bytes):
            raise ObjectIntegrityError("stored bundle member differs from its manifest")
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if before_copy is not None:
            before_copy(info)
        materialize_object(store, info, path)


class AnalysisService:
    def __init__(
        self,
        repository: DatabaseRepository,
        store: ObjectStore,
        registry: ArchitectureRegistry,
        work_root: Path,
        *,
        max_pending: int = 8,
        heartbeat_seconds: float = 10.0,
        cache_budget_bytes: int = 512 * 1024**2,
        time_budget_seconds: int = 120,
        memory_budget_bytes: int = 4 * 1024**3,
        scratch_budget_bytes: int = 128 * 1024**2,
    ) -> None:
        if (
            max_pending < 1
            or not math.isfinite(heartbeat_seconds)
            or heartbeat_seconds <= 0
            or cache_budget_bytes < 1
            or any(
                type(value) is not int or value < 1
                for value in (time_budget_seconds, memory_budget_bytes, scratch_budget_bytes)
            )
        ):
            raise ValueError("pending quota and heartbeat interval must be positive")
        self.repository = repository
        self.store = store
        self.registry = registry
        self.work_root = work_root
        self.work_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.max_pending = max_pending
        self.heartbeat_seconds = heartbeat_seconds
        self.cache_budget_bytes = cache_budget_bytes
        self.time_budget_seconds = time_budget_seconds
        self.memory_budget_bytes = memory_budget_bytes
        self.scratch_budget_bytes = scratch_budget_bytes

    def authenticate(self, raw_token: str) -> Principal:
        return self.repository.authenticate(raw_token)

    def readiness(self) -> dict[str, JsonValue]:
        database = self.repository.readiness()
        return {
            "database": database,
            "active_model": database and self.repository.get_active_model() is not None,
        }

    def create_snapshot(self, principal: Principal, export: ExportV2) -> SnapshotRecord:
        export = ExportV2.model_validate_json(export.model_dump_json())
        if (
            principal.role == "integration"
            and export.source.source_id != principal.source_namespace
        ):
            raise Forbidden("The snapshot source differs from this integration credential.")
        if principal.run_scope is not None and export.source.run_id != principal.run_scope:
            raise Forbidden("The snapshot run differs from this integration's asserted run.")
        data = json_bytes(export.model_dump(mode="json"))
        if len(data) > MAX_INPUT_BYTES:
            raise ValueError("export exceeds the 16 MiB input size limit")
        snapshot_id = uuid4()
        data_sha256 = hashlib.sha256(data).hexdigest()
        owner_key = hashlib.sha256(json_bytes(principal.owner_id)).hexdigest()
        with tempfile.TemporaryDirectory(prefix="snapshot-", dir=self.work_root) as temporary:
            path = Path(temporary) / "export.json"
            path.write_bytes(data)
            try:
                info = self.store.put(
                    f"snapshots/{owner_key}/{data_sha256}.json",
                    path,
                    expected_sha256=data_sha256,
                )
            except ObjectCapacityError as exc:
                raise CapacityExceeded("Snapshot storage capacity is exhausted.") from exc
        record = SnapshotRecord(
            snapshot_id=snapshot_id,
            owner_id=principal.owner_id,
            source_id=export.source.source_id,
            run_id=export.source.run_id,
            source_snapshot_id=export.source.snapshot_id,
            content_sha256=export.content_sha256,
            object=info,
            created_at=datetime.now(UTC),
        )
        # A failed database commit leaves an unreferenced immutable object. Cleanup
        # requires a separate authorized reconciliation with reference checks.
        return self.repository.add_snapshot(record)

    def get_snapshot(self, principal: Principal, snapshot_id: UUID) -> SnapshotRecord:
        return self.repository.get_snapshot(principal, snapshot_id)

    def read_snapshot(self, principal: Principal, snapshot_id: UUID) -> ExportV2:
        record = self.get_snapshot(principal, snapshot_id)
        return self._read_snapshot_record(record)

    def _read_snapshot_record(self, record: SnapshotRecord) -> ExportV2:
        if record.object.size_bytes > MAX_INPUT_BYTES:
            raise ObjectIntegrityError("stored snapshot exceeds the input size limit")
        with tempfile.TemporaryDirectory(prefix="snapshot-read-", dir=self.work_root) as temporary:
            path = Path(temporary) / "export.json"
            materialize_object(self.store, record.object, path)
            export = parse_export_v2(path.read_bytes())
        if (
            export.content_sha256,
            export.source.source_id,
            export.source.run_id,
            export.source.snapshot_id,
        ) != (
            record.content_sha256,
            record.source_id,
            record.run_id,
            record.source_snapshot_id,
        ):
            raise ObjectIntegrityError("stored snapshot differs from its committed identity")
        return export

    def submit_analysis(
        self,
        principal: Principal,
        snapshot_id: UUID,
        *,
        model_version_id: str = "active",
        idempotency_key: str,
    ) -> AnalysisRecord:
        # Hash the requested alias, not today's active resolution: a replay must
        # retain the immutable model selected by the first accepted request.
        request_sha256 = hashlib.sha256(
            json_bytes({"snapshot_id": str(snapshot_id), "model_version_id": model_version_id})
        ).hexdigest()
        return self.repository.submit_analysis(
            principal,
            snapshot_id,
            model_version_id=model_version_id,
            idempotency_key=idempotency_key,
            request_sha256=request_sha256,
            max_pending=self.max_pending,
        )

    def list_analyses(
        self, principal: Principal, *, limit: int = 100, offset: int = 0
    ) -> tuple[AnalysisRecord, ...]:
        return self.repository.list_analyses(principal, limit=limit, offset=offset)

    def get_analysis(self, principal: Principal, analysis_id: UUID) -> AnalysisRecord:
        return self.repository.get_analysis(principal, analysis_id)

    def get_analysis_summary(self, principal: Principal, analysis_id: UUID) -> AnalysisSummary:
        return self.repository.get_analysis_summary(principal, analysis_id)

    def list_analysis_summaries(
        self, principal: Principal, *, limit: int = 100, offset: int = 0
    ) -> tuple[AnalysisSummary, ...]:
        return self.repository.list_analysis_summaries(principal, limit=limit, offset=offset)

    def get_result(self, principal: Principal, analysis_id: UUID) -> dict[str, JsonValue]:
        record = self.get_analysis(principal, analysis_id)
        if record.operation.state != "succeeded" or record.result is None:
            raise Conflict("The analysis has no committed successful result.")
        if record.result.get("result_version") != "2.0":
            raise UnsupportedContract()
        result = AnalysisResult.model_validate_json(json_bytes(record.result))
        snapshot = self.get_snapshot(principal, record.snapshot_id)
        if (
            result.analysis_id != analysis_id
            or result.snapshot_id != record.snapshot_id
            or hashlib.sha256(json_bytes(record.result)).hexdigest()
            != record.operation.result_sha256
            or result.snapshot_sha256 != snapshot.content_sha256
            or result.model != record.operation.model
            or (result.source.source_id, result.source.run_id, result.source.snapshot_id)
            != (snapshot.source_id, snapshot.run_id, snapshot.source_snapshot_id)
        ):
            raise ObjectIntegrityError("analysis result identity differs from its resource")
        return result.model_dump(mode="json")

    def get_report_metadata(
        self, principal: Principal, analysis_id: UUID
    ) -> AnalysisReportRecord | None:
        summary = self.get_analysis_summary(principal, analysis_id)
        report = self.repository.get_analysis_report(principal, analysis_id)
        if report is None:
            return None
        snapshot = self.get_snapshot(principal, summary.snapshot_id)
        if (
            summary.operation.state != "succeeded"
            or report.analysis_id != analysis_id
            or report.owner_id != principal.owner_id
            or report.snapshot_id != snapshot.snapshot_id
            or report.snapshot_sha256 != snapshot.content_sha256
            or report.result_sha256 != summary.operation.result_sha256
            or report.model != summary.operation.model
            or (report.source.source_id, report.source.run_id, report.source.snapshot_id)
            != (snapshot.source_id, snapshot.run_id, snapshot.source_snapshot_id)
            or report.object.size_bytes > MAX_REPORT_BYTES
        ):
            raise ObjectIntegrityError("report identity differs from its committed analysis")
        return report

    def read_report(self, principal: Principal, analysis_id: UUID) -> bytes:
        # Authorization precedes object access, including integration source/run
        # checks. Reading a report never starts rendering or a scientific task.
        summary = self.get_analysis_summary(principal, analysis_id)
        if summary.operation.state != "succeeded":
            raise Conflict("The analysis has no committed successful report.")
        report = self.get_report_metadata(principal, analysis_id)
        if report is None:
            raise NotFound("This saved analysis has no available offline report.")
        result = self.get_result(principal, analysis_id)
        if result["source"] != report.source.model_dump(mode="json"):
            raise ObjectIntegrityError("report source differs from its saved result")
        with tempfile.TemporaryDirectory(prefix="report-read-", dir=self.work_root) as temporary:
            path = Path(temporary) / "report.html"
            materialize_object(self.store, report.object, path)
            payload = path.read_bytes()
        if len(payload) > MAX_REPORT_BYTES:
            raise ObjectIntegrityError("the stored report exceeds its download limit")
        return payload

    def get_operation(self, principal: Principal, operation_id: UUID) -> Operation:
        return self.repository.get_operation(principal, operation_id)

    def cancel_operation(self, principal: Principal, operation_id: UUID) -> Operation:
        return self.repository.cancel_operation(principal, operation_id)

    def list_models(
        self, principal: Principal, *, limit: int = 100, offset: int = 0
    ) -> tuple[ModelRecord, ...]:
        return self.repository.list_models(principal, limit=limit, offset=offset)

    def get_model(self, principal: Principal, model_version_id: str) -> ModelRecord:
        return self.repository.get_model_for_principal(principal, model_version_id)

    def activate_model(
        self, principal: Principal, model_version_id: str, *, expected_previous: str | None
    ) -> ModelRecord:
        if principal.role != "admin":
            raise Forbidden()
        model = self.repository.get_model(model_version_id)
        # A verified local cache can outlive lost or corrupted authoritative
        # objects. Publication verifies those objects remain restorable as well.
        for info in (model.manifest, *model.members.values()):
            if self.store.stat(info.key) != info:
                raise ObjectIntegrityError("model storage differs from its committed generation")
        from regimex.application.analysis import load_registered_predictor

        load_registered_predictor(self, model, worker_id="publication")
        return self.repository.activate_model(
            principal, model_version_id, expected_previous=expected_previous
        )

    def list_architectures(self, principal: Principal) -> tuple[ArchitectureInfo, ...]:
        if principal.role != "admin":
            raise Forbidden()
        return tuple(descriptor.describe() for descriptor in self.registry.descriptors)

    def register_bundle(
        self,
        principal: Principal,
        root: Path,
        *,
        manifest_sha256: str,
        inference_runtime: InferenceRuntime,
        activate: bool = False,
        expected_previous: str | None = None,
        worker_id: str = "registration",
    ) -> ModelRecord:
        """Explicit administrator bootstrap; verify actual saved backend restoration.

        This operational path does not turn validation evidence into independent
        quality evidence. Full candidate/evaluation publication is separate.
        """
        if principal.role != "admin":
            raise Forbidden()
        manifest_path = root / "manifest.json"
        root_descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            descriptor_fd = os.open(
                "manifest.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root_descriptor
            )
            with os.fdopen(descriptor_fd, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError("bundle manifest must be a regular file")
                data = stream.read(MAX_BUNDLE_MANIFEST_BYTES + 1)
        finally:
            os.close(root_descriptor)
        if len(data) > MAX_BUNDLE_MANIFEST_BYTES:
            raise ValueError("bundle manifest exceeds its input size limit")
        manifest = ModelBundle.model_validate_json(data)
        descriptor = self.registry.get(manifest.architecture_id)
        manifest = verify_model_bundle(
            root,
            manifest_sha256=manifest_sha256,
            architecture_id=descriptor.architecture_id,
            implementation_version=descriptor.implementation_version,
        )
        total_size = len(data) + sum(member.size_bytes for member in manifest.members)
        if total_size > inference_runtime.model_cache_budget_bytes:
            raise CapacityExceeded("Model bundle exceeds the inference cache budget.")
        policy_sha256 = hashlib.sha256(
            json_bytes(inference_runtime.model_dump(mode="json"))
        ).hexdigest()
        try:
            existing = self.repository.get_model(manifest.model_version_id)
        except NotFound:
            existing = None
        if existing is not None:
            if (existing.bundle_sha256, existing.inference_policy_sha256) != (
                manifest_sha256,
                policy_sha256,
            ):
                raise Conflict("The model version already has another immutable bundle or policy.")
            from regimex.application.analysis import load_registered_predictor

            load_registered_predictor(self, existing, worker_id=worker_id)
            if activate:
                return self.activate_model(
                    principal, existing.model_version_id, expected_previous=expected_previous
                )
            return existing
        prefix = f"models/{uuid4()}"
        manifest_object = self.store.put(
            f"{prefix}/manifest.json", manifest_path, expected_sha256=manifest_sha256
        )
        members = {
            member.path: self.store.put(
                f"{prefix}/{member.path}", root / member.path, expected_sha256=member.sha256
            )
            for member in manifest.members
        }
        record = ModelRecord(
            model_version_id=manifest.model_version_id,
            architecture_id=manifest.architecture_id,
            bundle_sha256=manifest_sha256,
            manifest=manifest_object,
            members=members,
            inference_runtime=inference_runtime,
            inference_policy_sha256=policy_sha256,
            created_at=datetime.now(UTC),
        )
        # Imported only in the dedicated operational computation process.
        from regimex.application.analysis import load_registered_predictor

        load_registered_predictor(self, record, worker_id=worker_id)
        self.repository.register_model(principal, record)
        if activate:
            return self.activate_model(
                principal, record.model_version_id, expected_previous=expected_previous
            )
        return record

    def run_analysis(self, operation_id: UUID, *, worker_id: str = "analysis") -> Operation | None:
        from regimex.application.analysis import run_analysis

        return run_analysis(self, operation_id, worker_id=worker_id)
