"""A run keeps its selection evidence even when the history cannot be previewed."""

from pathlib import Path

import pytest
from pydantic import JsonValue
from test_model_bundles import bundle

from regimex.admin_contracts import ImportedTrainingSpec, RepeatOutcome, TrainingSpec
from regimex.application.training_metrics import build_training_metrics
from regimex.contracts import MAX_INPUT_BYTES
from regimex.infrastructure.object_store import ObjectIntegrityError
from regimex.model_bundles import BundleMember, ModelBundle
from regimex.model_registry import (
    FixedUpdatePolicy,
    InferenceRuntime,
    TrainingPolicy,
    TrainingRuntime,
)
from regimex.storage import json_bytes


@pytest.fixture
def spec() -> TrainingSpec:
    return TrainingSpec(
        dataset_id="dataset-1",
        labels_reference_set_id="train-1",
        validation_reference_set_id="validation-1",
        model_version_id="baseline-1",
        architecture_id="baseline",
        parameters={},
        policy=TrainingPolicy(optimizer="lbfgs", seeds=(11, 23)),
        training_runtime=TrainingRuntime(),
        inference_runtime=InferenceRuntime(),
        analysis_profile_id="direction-32/2.0",
        build_id="build-1",
    )


def manifest_with_history(root: Path, path: str, size: int) -> ModelBundle:
    manifest = bundle(root)
    members = tuple(
        member.model_copy(update={"path": path, "size_bytes": size})
        if member.role == "history"
        else member
        for member in manifest.members
    )
    return manifest.model_copy(update={"members": members})


@pytest.mark.parametrize(
    ("path", "size"), [("history.bin", 10), ("history.json", MAX_INPUT_BYTES + 1)]
)
def test_unavailable_history_keeps_summary_without_reading_member(
    tmp_path: Path, spec: TrainingSpec, path: str, size: int
) -> None:
    manifest = manifest_with_history(tmp_path / "bundle", path, size)

    def unexpected_read(member: BundleMember) -> bytes:
        pytest.fail(f"Attempted to load {member.path}")

    metrics = build_training_metrics(spec, manifest, read_history=unexpected_read)
    assert metrics.selected_seed == manifest.training.selected_seed
    assert metrics.validation_segment_macro_f1 == manifest.training.validation_segment_macro_f1
    assert metrics.history is None and metrics.history_note
    assert metrics.repeat_outcomes == ()


@pytest.mark.parametrize("invalid", [b'{"score": NaN}', b'{"score": 1e999}', b"invalid"])
def test_invalid_history_cannot_be_committed(
    tmp_path: Path, spec: TrainingSpec, invalid: bytes
) -> None:
    manifest = manifest_with_history(tmp_path / "bundle", "history.json", len(invalid))
    with pytest.raises(ObjectIntegrityError, match="history is invalid JSON"):
        build_training_metrics(spec, manifest, read_history=lambda _: invalid)


def imported_spec(
    manifest: ModelBundle, architecture: str, policy: TrainingPolicy | FixedUpdatePolicy
) -> ImportedTrainingSpec:
    return ImportedTrainingSpec(
        origin="external",
        architecture_id=architecture,
        model_version_id=manifest.model_version_id,
        build_id=manifest.build_id,
        implementation_version=manifest.implementation_version,
        analysis_profile_id=manifest.analysis_profile,
        evidence=manifest.training,
        weights=next(member for member in manifest.members if member.role == "weights"),
        inference_runtime=InferenceRuntime(),
        configuration={"training_policy": policy.model_dump(mode="json")},
    )


@pytest.mark.parametrize("architecture", ("tcn", "segment-moe", "persistence"))
def test_imported_epoch_history_records_each_repeats_stopping_reason(
    tmp_path: Path, architecture: str
) -> None:
    history: dict[str, JsonValue] = {
        "epochs": [
            {"seed": 11, "epoch": 1},
            {"seed": 11, "epoch": 2},
            {"seed": 23, "epoch": 1},
            {"seed": 23, "epoch": 2},
            {"seed": 23, "epoch": 3},
        ]
    }
    payload = json_bytes(history)
    manifest = manifest_with_history(tmp_path / "bundle", "history.json", len(payload))
    policy = TrainingPolicy(optimizer="adamw", seeds=(11, 23), epochs=3, patience=1)
    spec = imported_spec(manifest, architecture, policy)
    metrics = build_training_metrics(spec, manifest, read_history=lambda _: payload)
    assert metrics.repeat_outcomes == (
        RepeatOutcome(seed=11, step=2, reason="early_stopping"),
        RepeatOutcome(seed=23, step=3, reason="epoch_limit"),
    )
    assert not RepeatOutcome.from_history(spec.model_copy(update={"configuration": None}), history)


def test_imported_update_history_records_the_update_limit(tmp_path: Path) -> None:
    manifest = bundle(tmp_path / "bundle")
    policy = FixedUpdatePolicy(seeds=(11, 23), updates=1, validation_updates=(1,))
    spec = imported_spec(manifest, "tcn", policy)
    history: dict[str, JsonValue] = {
        "updates": [{"seed": 11, "update": 1}, {"seed": 23, "update": 1}]
    }
    assert RepeatOutcome.from_history(spec, history) == (
        RepeatOutcome(seed=11, step=1, reason="update_limit"),
        RepeatOutcome(seed=23, step=1, reason="update_limit"),
    )
