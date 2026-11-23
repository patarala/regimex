"""Retrospective validation enriches a completed forest without rewriting its fit."""

import json
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest
from pydantic import JsonValue
from test_adaptive_backend import fitted
from test_admin_workflow import admin as admin
from test_admin_workflow import completed, prepared_references
from test_baseline_backend import provenance, rehash

from regimex.admin_contracts import (
    AdminRunRecord,
    ForestValidationCurve,
    ForestValidationPoint,
    TrainingSpec,
)
from regimex.application.admin_results import AdminResultCommitter
from regimex.application.admin_service import AdminService
from regimex.application.service import materialize_bundle
from regimex.backends.adaptive import AdaptiveBackend
from regimex.directions import DEFAULT_ANALYSIS_PROFILE
from regimex.model_registry import InferenceRuntime, SingleFitPolicy, TrainingRuntime
from regimex.service_contracts import Conflict, ModelRecord, Principal
from regimex.storage import digest, json_bytes, write_json


def forest_import(
    service: AdminService, principal: Principal, root: Path, *, final_point_only: bool = True
) -> tuple[ForestValidationCurve, partial[ModelRecord]]:
    result, _, _ = fitted()
    original = replace(result, history=result.history[-1:]) if final_point_only else result
    origin = provenance(original)
    evidence = origin.training.model_copy(update={"validation_dataset_sha256": "a" * 64})
    origin = origin.model_copy(update={"model_version_id": "fixture-forest", "training": evidence})
    sha = AdaptiveBackend().write_bundle(
        original, root, provenance=origin, analysis=original.predictor.analysis
    )
    curve = ForestValidationCurve(
        bundle_sha256=sha,
        dataset_sha256="a" * 64,
        validation_reference_sha256=evidence.validation_references.content_sha256,
        metric=evidence.selection_metric,
        points=tuple(
            ForestValidationPoint.model_validate(
                {
                    "trees": row["trees"],
                    "validation_segment_macro_f1": row["validation_segment_macro_f1"],
                }
            )
            for row in result.history
        ),
    )
    return curve, partial(
        service.analysis_service.register_bundle,
        principal,
        root,
        manifest_sha256=sha,
        inference_runtime=InferenceRuntime(),
        record_training=True,
    )


def other_score(point: ForestValidationPoint) -> ForestValidationPoint:
    changed = 1.0 if point.validation_segment_macro_f1 == 0 else 0.0
    return point.model_copy(update={"validation_segment_macro_f1": changed})


def test_reimport_adds_one_curve_to_the_same_run(
    admin: tuple[AdminService, Principal], tmp_path: Path
) -> None:
    service, principal = admin
    root = tmp_path / "forest"
    curve, register = forest_import(service, principal, root)
    model = register(display_name="Original forest")
    (original_run,) = service.list_runs(principal, kind="training")
    original = service.training_metrics(principal, original_run.run_id)
    assert json_bytes(original.history) == (root / "history.json").read_bytes()
    assert original.forest_validation_curve is None

    assert register(forest_validation_curve=curve) == model
    assert register(forest_validation_curve=curve) == model
    assert register() == model
    assert service.list_runs(principal, kind="training") == (original_run,)
    saved = service.training_metrics(principal, original_run.run_id)
    assert saved.forest_validation_curve == curve
    assert json_bytes(
        saved.model_dump(mode="json", exclude={"forest_validation_curve"})
    ) == json_bytes(original.model_dump(mode="json"))

    changed = curve.model_copy(update={"points": (other_score(curve.points[0]), *curve.points[1:])})
    with pytest.raises(Conflict, match="different validation curve"):
        register(forest_validation_curve=changed)
    assert service.training_metrics(principal, original_run.run_id) == saved
    assert digest(root / "manifest.json") == model.bundle_sha256


def test_first_import_requires_matching_evidence_before_creating_a_run(
    admin: tuple[AdminService, Principal], tmp_path: Path
) -> None:
    service, principal = admin
    root = tmp_path / "forest"
    curve, register = forest_import(service, principal, root)
    last = curve.points[-1]
    for change in (
        {"bundle_sha256": "0" * 64},
        {"dataset_sha256": "3" * 64},  # The fixture's training and validation datasets differ.
        {"validation_reference_sha256": "0" * 64},
        {"points": (*curve.points[:-1], other_score(last))},
        {"points": (*curve.points[:-1], last.model_copy(update={"trees": last.trees + 1}))},
    ):
        with pytest.raises(Conflict, match="original single fit"):
            register(forest_validation_curve=curve.model_copy(update=change))
    assert service.list_runs(principal, kind="training") == ()

    register(forest_validation_curve=curve)
    (run,) = service.list_runs(principal, kind="training")
    saved = service.training_metrics(principal, run.run_id)
    assert saved.forest_validation_curve == curve
    assert json_bytes(saved.history) == (root / "history.json").read_bytes()


def test_an_existing_prefix_history_cannot_be_replaced_by_a_retrospective_curve(
    admin: tuple[AdminService, Principal], tmp_path: Path
) -> None:
    service, principal = admin
    root = tmp_path / "forest"
    curve, register = forest_import(service, principal, root, final_point_only=False)
    register()
    (run,) = service.list_runs(principal, kind="training")
    with pytest.raises(Conflict, match="original single fit"):
        register(forest_validation_curve=curve)
    original = service.training_metrics(principal, run.run_id)
    assert json_bytes(original.history) == (root / "history.json").read_bytes()
    assert original.forest_validation_curve is None


def test_attach_curve_to_worker_run_preserves_training_request_and_statistics(
    admin: tuple[AdminService, Principal], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, principal = admin
    dataset, validation = prepared_references(service, principal)
    submitted = service.submit_training(
        principal,
        TrainingSpec(
            dataset_id=dataset.dataset_id,
            labels_reference_set_id="fixture-labels",
            validation_reference_set_id=validation.references.reference_set_id,
            model_version_id="worker-forest",
            architecture_id="adaptive",
            parameters={"trees": 3, "max_depth": 3},
            policy=SingleFitPolicy(optimizer="extra_trees", seeds=(11,)),
            training_runtime=TrainingRuntime(threads=1, time_budget_seconds=120),
            inference_runtime=InferenceRuntime(threads=1),
            analysis_profile_id=DEFAULT_ANALYSIS_PROFILE,
            build_id="fixture-forest",
            benchmark_selection_id="fixture-final-benchmark",
        ),
        idempotency_key="worker-forest",
    )
    commit = AdminResultCommitter.commit
    points: list[ForestValidationPoint] = []

    def commit_original_bundle(
        committer: AdminResultCommitter, child_result: dict[str, JsonValue]
    ) -> AdminRunRecord:
        # Earlier workers committed only the full-forest score in the bundle.
        root = committer.root / "output/bundle"
        rows = json.loads((root / "history.json").read_bytes())["fits"]
        for row in rows:
            points.append(
                ForestValidationPoint(
                    trees=row["trees"],
                    validation_segment_macro_f1=row["validation_segment_macro_f1"],
                )
            )
        write_json(root / "history.json", {"fits": rows[-1:]})
        candidate = child_result["candidate"]
        assert isinstance(candidate, dict)
        candidate["manifest_sha256"] = rehash(root)
        return commit(committer, child_result)

    with monkeypatch.context() as patch:
        patch.setattr(AdminResultCommitter, "commit", commit_original_bundle)
        completed(service, submitted.run_id)
    original = service.get_run(principal, submitted.run_id)
    assert original.output and original.output["fit_statistics"]
    model = service.analysis_service.repository.get_model("worker-forest")
    root = tmp_path / "worker-forest"
    materialize_bundle(service.analysis_service.store, model, root)
    curve = ForestValidationCurve(
        bundle_sha256=model.bundle_sha256,
        dataset_sha256=dataset.manifest.content_sha256,
        validation_reference_sha256=validation.references.content_sha256,
        metric="decoded_segment_macro_f1/iou-0.50",
        points=tuple(points),
    )
    service.analysis_service.register_bundle(
        principal,
        root,
        manifest_sha256=model.bundle_sha256,
        inference_runtime=model.inference_runtime,
        record_training=True,
        forest_validation_curve=curve,
    )
    assert service.list_runs(principal, kind="training") == (original,)
    assert original.spec == submitted.spec
    metrics = service.training_metrics(principal, submitted.run_id)
    assert json_bytes(metrics.history) == (root / "history.json").read_bytes()
    assert metrics.forest_validation_curve == curve
