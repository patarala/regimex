"""A run keeps its selection evidence even when the history cannot be previewed."""

from pathlib import Path

import pytest
from test_model_bundles import bundle

from regimex.admin_contracts import TrainingSpec
from regimex.application.training_metrics import build_training_metrics
from regimex.contracts import MAX_INPUT_BYTES
from regimex.infrastructure.object_store import ObjectIntegrityError
from regimex.model_bundles import BundleMember, ModelBundle
from regimex.model_registry import InferenceRuntime, TrainingPolicy, TrainingRuntime


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
