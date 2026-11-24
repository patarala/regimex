"""Read training evidence from the verified, registered object generations."""

import tempfile
from pathlib import Path

from pydantic import JsonValue, RootModel

from regimex.admin_contracts import ForestValidationCurve, ImportedTrainingSpec
from regimex.application.service import AnalysisService, materialize_object
from regimex.application.training_metrics import build_training_metrics
from regimex.direction_quality import DirectionQualityEvidence
from regimex.external_evaluations import ExternalEvaluation
from regimex.external_latency import ExternalLatencyEvidence
from regimex.infrastructure.admin_repository import AdminRepository
from regimex.model_bundles import BundleMember, ModelBundle
from regimex.operations import LaunchSource
from regimex.service_contracts import ModelRecord, Principal


def record_external_training(
    service: AnalysisService,
    principal: Principal,
    model: ModelRecord,
    manifest: ModelBundle,
    display_name: str | None,
    evaluations: tuple[ExternalEvaluation, ...],
    latency: ExternalLatencyEvidence | None,
    launch_source: LaunchSource | None,
    forest_validation_curve: ForestValidationCurve | None,
    direction_quality: DirectionQualityEvidence | None,
) -> ModelRecord:
    members = {member.role: member for member in manifest.members}
    with tempfile.TemporaryDirectory(dir=service.work_root) as tmp:

        def read_member(member: BundleMember) -> bytes:
            destination = Path(tmp) / member.role
            materialize_object(service.store, model.members[member.path], destination)
            return destination.read_bytes()

        configuration = members["configuration"]
        settings = None
        if configuration.path.endswith(".json") and configuration.size_bytes <= 64 * 1024:
            content = RootModel[JsonValue].model_validate_json(read_member(configuration)).root
            if isinstance(content, dict):
                settings = content
        spec = ImportedTrainingSpec(
            origin="external",
            model_version_id=model.model_version_id,
            architecture_id=model.architecture_id,
            implementation_version=manifest.implementation_version,
            build_id=manifest.build_id,
            analysis_profile_id=manifest.analysis_profile,
            evidence=manifest.training,
            weights=members["weights"],
            configuration=settings,
            inference_runtime=model.inference_runtime,
        )
        metrics = build_training_metrics(spec, manifest, read_history=read_member)
        metrics = metrics.model_copy(
            update={
                "forest_validation_curve": forest_validation_curve,
                "direction_quality": direction_quality,
            }
        )
    # The model and its completed history become visible together, after all byte verification.
    return AdminRepository(service.repository).register_external_training(
        principal,
        model,
        spec,
        metrics,
        display_name=display_name,
        evaluations=evaluations,
        latency=latency,
        launch_source=launch_source,
    )
