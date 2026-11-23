"""Exercise adaptive forest training, publication, resource limits and original-grid inference."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import JsonValue

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from test_application_training import prepared_data, request
from test_baseline_backend import inputs, provenance, rehash

from regimex.application.training import (
    SharedSelectionEvaluator,
    observation_inputs,
    train_candidate,
)
from regimex.backends.adaptive import AdaptiveBackend, AdaptivePredictor
from regimex.backends.adaptive_forest import features, fit
from regimex.directions import AnalysisConfig
from regimex.model_contracts import FitResult, ObservationInputs, TrainingInputs
from regimex.model_registry import (
    AdaptiveParameters,
    ArchitectureRegistry,
    InferenceRuntime,
    SingleFitPolicy,
    TrainingPolicy,
    TrainingRuntime,
    adaptive_descriptor,
)
from regimex.resource_limits import ResourceBudgetExceeded
from regimex.storage import write_json


def fitted() -> tuple[FitResult, ObservationInputs, TrainingInputs]:
    train, valid = inputs("train"), inputs("validation")
    y = np.repeat(np.arange(3), (15, 21, 18))
    y[::11] = -1
    data = TrainingInputs(train, y)
    fitted = AdaptiveBackend().fit(
        data,
        valid,
        parameters=AdaptiveParameters(trees=3, max_depth=3),
        policy=SingleFitPolicy(optimizer="extra_trees", seeds=(11,)),
        analysis=AnalysisConfig(min_duration=4),
        runtime=TrainingRuntime(threads=1),
        evaluator=lambda p: float(p[:, 0].mean()),
    )
    return fitted, valid, data


def test_only_training_rows_enter_the_forest_and_every_instrument_keeps_its_grid() -> None:
    result, valid, training = fitted()
    predictor = result.predictor
    assert isinstance(predictor, AdaptivePredictor)
    model = fit(
        np.concatenate(
            [features(rows, predictor.analysis) for rows in training.observations.observations]
        ),
        training.targets,
        trees=3,
        max_depth=3,
        seed=11,
    )
    expected = np.concatenate(
        [model.predict(features(rows, predictor.analysis)) for rows in valid.observations]
    )
    for limit in (1, 8):
        actual = predictor.predict(valid, runtime=InferenceRuntime(max_batch_size=limit))
        np.testing.assert_array_equal(actual, expected)
    assert result.selected_settings == {"trees": 3, "max_depth": 3}
    assert result.selected_seed == 11
    assert result.validation_score == pytest.approx(float(expected[:, 0].mean()))


def test_registered_backend_fits_scores_and_restores_a_candidate(tmp_path: Path) -> None:
    data, _ = prepared_data()
    intake = request().model_copy(
        update={
            "architecture_id": "adaptive",
            "parameters": {"trees": 5, "max_depth": 4},
            "policy": SingleFitPolicy(optimizer="extra_trees", seeds=(35,)),
        }
    )
    events: list[dict[str, JsonValue]] = []
    root = tmp_path / "candidate"
    result = train_candidate(
        data,
        intake,
        root,
        registry=ArchitectureRegistry((adaptive_descriptor(),)),
        worker_id="adaptive-test",
        on_history=events.append,
    )
    saved = json.loads((root / "history.json").read_bytes())
    assert saved == {"fits": events}
    assert events == [
        {
            "trees": 5,
            "max_depth": 4,
            "seed": 35,
            "validation_segment_macro_f1": result.validation_segment_macro_f1,
        }
    ]
    predictor = AdaptiveBackend().load_bundle(
        root, manifest_sha256=result.manifest_sha256, runtime=intake.verification_runtime
    )
    valid = observation_inputs(data.validation)
    evaluator = SharedSelectionEvaluator(
        valid, (data.validation[0].annotations.intervals,), data.analysis_profile, "human"
    )
    assert (
        evaluator(predictor.predict(valid, runtime=intake.verification_runtime))
        == result.validation_segment_macro_f1
    )


def test_cache_budget_covers_expanded_weights_before_load_and_predict(tmp_path: Path) -> None:
    result, valid, _ = fitted()
    backend = AdaptiveBackend()
    sha = backend.write_bundle(
        result, tmp_path, provenance=provenance(result), analysis=result.predictor.analysis
    )
    assert isinstance(result.predictor, AdaptivePredictor)
    budget = result.predictor.numeric_bytes
    with pytest.raises(ResourceBudgetExceeded):
        backend.load_bundle(
            tmp_path,
            manifest_sha256=sha,
            runtime=InferenceRuntime(model_cache_budget_bytes=budget - 1),
        )
    predictor = backend.load_bundle(
        tmp_path, manifest_sha256=sha, runtime=InferenceRuntime(model_cache_budget_bytes=budget)
    )
    with pytest.raises(ResourceBudgetExceeded):
        predictor.predict(valid, runtime=InferenceRuntime(model_cache_budget_bytes=budget - 1))
    with pytest.raises(ValueError, match="committed digest"):
        backend.load_bundle(tmp_path, manifest_sha256="0" * 64, runtime=InferenceRuntime())


@pytest.mark.parametrize(
    "member", ["history", "preprocessing", "validation", "configuration", "weights"]
)
def test_rejects_resealed_incompatible_bundle_members(tmp_path: Path, member: str) -> None:
    result, _, _ = fitted()
    backend = AdaptiveBackend()
    backend.write_bundle(
        result, tmp_path, provenance=provenance(result), analysis=result.predictor.analysis
    )
    if member == "weights":
        # A valid archive from a different forest must not satisfy the recorded tree count.
        with np.load(tmp_path / "weights.npz", allow_pickle=False) as arrays:
            values = dict(arrays)
        values["offsets"] = values["offsets"][:1]
        np.savez(tmp_path / "weights.npz", **values)
    else:
        path = tmp_path / (member + ".json")
        changed = json.loads(path.read_bytes())
        if member == "history":
            changed["fits"][0]["seed"] += 1
        elif member == "validation":
            changed["score"] = 1 - changed["score"]
        elif member == "configuration":
            changed["codec_version"] = "unknown/99"
        else:
            changed["feature_names"].reverse()
        write_json(path, changed)
    sha = rehash(tmp_path)
    with pytest.raises(ValueError):
        backend.load_bundle(tmp_path, manifest_sha256=sha, runtime=InferenceRuntime())


def test_fit_and_publication_refuse_inconsistent_inputs(tmp_path: Path) -> None:
    backend = AdaptiveBackend()
    result, valid, training = fitted()
    predictor = result.predictor
    assert isinstance(predictor, AdaptivePredictor)
    config = predictor.configuration
    for validation, policy, runtime, error in (
        (training.observations, config.training_policy, config.training_runtime, ValueError),
        (valid, TrainingPolicy(optimizer="adamw"), config.training_runtime, ValueError),
        (
            valid,
            config.training_policy,
            TrainingRuntime(memory_budget_bytes=1),
            ResourceBudgetExceeded,
        ),
    ):
        with pytest.raises(error):
            backend.fit(
                training,
                validation,
                policy=policy,
                parameters=config.parameters,
                runtime=runtime,
                analysis=config.analysis,
                evaluator=lambda _: 0.5,
            )
    with pytest.raises(ValueError, match="selected settings"):
        backend.write_bundle(
            replace(result, selected_settings={"trees": 4, "max_depth": 3}),
            tmp_path / "output",
            provenance=provenance(result),
            analysis=config.analysis,
        )
    assert not (tmp_path / "output").exists()


def test_deadline_rejects_training_before_feature_allocation_exceeds_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from regimex.backends import adaptive

    train = inputs("train")
    valid = inputs("validation")
    clock = iter((0.0, 2.0))
    monkeypatch.setattr(adaptive, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    with pytest.raises(ResourceBudgetExceeded, match="time budget"):
        AdaptiveBackend().fit(
            TrainingInputs(train, np.repeat(np.arange(3), (15, 21, 18))),
            valid,
            parameters=AdaptiveParameters(trees=2, max_depth=2),
            policy=SingleFitPolicy(optimizer="extra_trees"),
            analysis=AnalysisConfig(min_duration=4),
            runtime=TrainingRuntime(time_budget_seconds=1),
            evaluator=lambda _: 0.5,
        )
