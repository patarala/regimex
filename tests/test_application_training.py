"""Registered training, independent validation and saved candidate verification."""

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from regimex.application.training import (
    SharedSelectionEvaluator,
    TrainingRequest,
    observation_inputs,
    train_candidate,
)
from regimex.backends.baseline import BaselineBackend
from regimex.benchmarks import DirectionSemantics
from regimex.data_catalog import (
    DatasetManifest,
    Partition,
    RawTrainingData,
    ReferencePayload,
    ReferenceSet,
    SourceSnapshot,
    SplitGroup,
    analysis_profile_hash,
    create_reference_member,
    create_snapshot_member,
    load_training_data,
    seal_dataset,
    seal_reference_set,
)
from regimex.directions import (
    AnalysisConfig,
    Direction,
    DirectionAnnotations,
    DirectionInterval,
    summarize_interval,
)
from regimex.exports import ExportPayloadV2, seal_export
from regimex.features import DoubleArray
from regimex.model_bundles import verify_model_bundle
from regimex.model_contracts import FitResult, LoadedPredictor
from regimex.model_registry import (
    ArchitectureRegistry,
    InferenceRuntime,
    TrainingPolicy,
    TrainingRuntime,
    baseline_descriptor,
)
from regimex.segmentation import DECODER_VERSION

PROFILE = AnalysisConfig()


def source(index: int) -> SourceSnapshot:
    # Deliberate direction fixtures test implementation, not simulator realism or quality.
    value = json.loads((Path(__file__).parent / "fixtures" / "experiment-v2.json").read_text())
    value.pop("content_sha256")
    identifier = f"fixture-{index}"
    for name in ("snapshot_id", "run_id", "configuration_id"):
        value["source"][name] = identifier
    value["producer"]["name"] = "training-behavior-fixture"
    value["time_grid"].update(origin=0.0, bin_width=1.0, completed_horizon=192.0, step_count=192)
    template = value["observations"][0]
    initial = 100.0 + index * 10
    value["instruments"][0]["initial_price"] = initial
    value["observations"] = [
        {
            **template,
            "step": step,
            "last_price": initial + (step if step < 64 else 128 - step if step < 128 else 0),
            "trade_count": 1,
            "trade_volume": 1.0,
            "price_state": "observed",
        }
        for step in range(192)
    ]
    return SourceSnapshot(
        source_namespace="behavior-fixtures",
        split_group=SplitGroup(namespace="fixture-seeds", group_id=identifier, seed=index),
        export=seal_export(ExportPayloadV2.model_validate_json(json.dumps(value))),
    )


def prepared_data() -> tuple[RawTrainingData, DatasetManifest]:
    sources = tuple(source(index) for index in range(3))
    partitions: tuple[Partition, ...] = ("train", "validation", "test")
    dataset = seal_dataset(
        "training-fixture",
        tuple(
            create_snapshot_member(item, partition)
            for item, partition in zip(sources, partitions, strict=True)
        ),
    )
    sets: list[ReferenceSet] = []
    documents: dict[str, DirectionAnnotations] = {}
    for member, item in zip(dataset.members[:2], sources[:2], strict=True):
        human = member.partition == "validation"
        rows = item.export.observations
        document = DirectionAnnotations(
            experiment_id=member.snapshot_id,
            policy="human-directions/1.0" if human else "retrospective-directions/3.0",
            policy_config=PROFILE.model_dump(mode="json"),
            reference_kind="human" if human else "automatic",
            reviewer="fixture-reviewer" if human else None,
            intervals=tuple(
                DirectionInterval(
                    instrument_id="A/Q",
                    start=start,
                    stop=stop,
                    direction=direction,
                    status="observed",
                    statistics=summarize_interval(rows, start, stop),
                )
                for start, stop, direction in (
                    (0, 64, Direction.GROWTH),
                    (64, 128, Direction.DECLINE),
                    (128, 192, Direction.SIDEWAYS),
                )
            ),
        )
        documents[member.snapshot_id] = document
        sets.append(
            seal_reference_set(
                ReferencePayload(
                    reference_set_id=f"{member.partition}-reference",
                    dataset_sha256=dataset.content_sha256,
                    partition=member.partition,
                    kind=document.reference_kind,
                    policy=document.policy,
                    semantics=DirectionSemantics(
                        profile_sha256=analysis_profile_hash(PROFILE),
                        decoder_version=DECODER_VERSION,
                    ),
                    analysis_profile=PROFILE,
                    reviewer=document.reviewer,
                    blind_review=human,
                    independent=human,
                    members=(create_reference_member(member, document),),
                )
            )
        )

    def read_snapshot(member: object) -> SourceSnapshot:
        for candidate, item in zip(dataset.members, sources, strict=True):
            if candidate == member:
                assert candidate.partition != "test", "training opened test observations"
                return item
        raise AssertionError("unknown snapshot request")

    def read_reference(reference: ReferenceSet, member: object) -> DirectionAnnotations:
        assert reference.partition != "test", "training opened final references"
        return documents[reference.members[0].snapshot_id]

    return load_training_data(
        dataset,
        sets[0],
        sets[1],
        snapshot_reader=read_snapshot,
        reference_reader=read_reference,
    ), dataset


def request() -> TrainingRequest:
    return TrainingRequest(
        run_id="fixture-run",
        model_version_id="candidate-1",
        architecture_id="baseline",
        parameters={"regularization": [1.0], "context_sizes": [4, 16]},
        policy=TrainingPolicy(optimizer="lbfgs", seeds=(11,)),
        training_runtime=TrainingRuntime(threads=1),
        verification_runtime=InferenceRuntime(threads=1),
        analysis_profile_id="direction-32/2.0",
        build_id="fixture-build",
    )


def test_training_uses_generic_registry_shared_selection_and_independent_references(
    tmp_path: Path,
) -> None:
    data, dataset = prepared_data()
    registry = ArchitectureRegistry((baseline_descriptor(),))
    result = train_candidate(
        data, request(), tmp_path / "candidate", registry=registry, worker_id="fixture-worker"
    )
    assert result.dataset_sha256 == dataset.content_sha256
    assert result.quality_status == "validation_only"
    bundle = verify_model_bundle(
        tmp_path / "candidate",
        manifest_sha256=result.manifest_sha256,
        architecture_id="baseline",
        implementation_version=baseline_descriptor().implementation_version,
    )
    assert bundle.training.labels.kind == "automatic"
    assert bundle.training.validation_references.kind == "human"
    assert (
        bundle.training.validation_references.content_sha256 == result.validation_reference_sha256
    )
    assert bundle.training.validation_segment_macro_f1 == result.validation_segment_macro_f1
    loaded = BaselineBackend().load_bundle(
        tmp_path / "candidate",
        manifest_sha256=result.manifest_sha256,
        runtime=request().verification_runtime,
    )
    inputs = observation_inputs(data.validation)
    probabilities = loaded.predict(inputs, runtime=request().verification_runtime)
    evaluator = SharedSelectionEvaluator(
        inputs, (data.validation[0].annotations.intervals,), PROFILE, "human"
    )
    assert evaluator(probabilities) == result.validation_segment_macro_f1
    assert not hasattr(inputs, "labels") and not hasattr(inputs.info[0], "seed")


@pytest.mark.parametrize("shape", [(191, 3), (192, 2)])
def test_shared_evaluator_rejects_lost_timestamps_and_wrong_class_shape(
    shape: tuple[int, ...],
) -> None:
    data, _ = prepared_data()
    evaluator = SharedSelectionEvaluator(
        observation_inputs(data.validation),
        (data.validation[0].annotations.intervals,),
        PROFILE,
        "human",
    )
    with pytest.raises(ValueError, match="invalid class scores"):
        evaluator(np.full(shape, 1 / 3, dtype=np.float64))


def test_failed_restoration_never_exposes_a_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, _ = prepared_data()

    def broken_load(*args: object, **kwargs: object) -> LoadedPredictor:
        raise ValueError("codec restoration failed")

    monkeypatch.setattr(BaselineBackend, "load_bundle", broken_load)
    with pytest.raises(ValueError, match="codec restoration failed"):
        train_candidate(
            data,
            request(),
            tmp_path / "candidate",
            registry=ArchitectureRegistry((baseline_descriptor(),)),
            worker_id="test",
        )
    assert not (tmp_path / "candidate").exists()
    assert not list(tmp_path.glob(".candidate-*"))


def test_unverified_backend_score_is_rejected_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, _ = prepared_data()
    original = BaselineBackend.fit

    def misreported_fit(*args: object, **kwargs: object) -> FitResult:
        fitted = original(*args, **kwargs)  # type: ignore[arg-type]
        return replace(fitted, validation_score=-0.5)

    monkeypatch.setattr(BaselineBackend, "fit", misreported_fit)
    with pytest.raises(ValueError, match="selection score differs"):
        train_candidate(
            data,
            request(),
            tmp_path / "candidate",
            registry=ArchitectureRegistry((baseline_descriptor(),)),
            worker_id="test",
        )
    assert not (tmp_path / "candidate").exists()


def test_evaluator_scores_the_shared_decoded_intervals() -> None:
    data, _ = prepared_data()
    evaluator = SharedSelectionEvaluator(
        observation_inputs(data.validation),
        (data.validation[0].annotations.intervals,),
        PROFILE,
        "human",
    )
    # Class order is sideways, growth, decline; every direction spans 64 steps.
    probabilities = np.full((192, 3), 0.001, dtype=np.float64)
    for start, stop, column in ((0, 64, 1), (64, 128, 2), (128, 192, 0)):
        probabilities[start:stop, column] = 0.998
    assert evaluator(cast(DoubleArray, probabilities)) == 1.0
    flat = np.full((192, 3), 0.001, dtype=np.float64)
    flat[:, 0] = 0.998
    assert evaluator(cast(DoubleArray, flat)) < 1.0
