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
from regimex.model_contracts import BundleProvenance, FitResult, ObservationInputs, TrainingInputs
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


def blended_bundle(
    tmp_path: Path,
) -> tuple[str, ObservationInputs, AdaptivePredictor, AdaptivePredictor]:
    from regimex.backends.adaptive_blend import blend_environment, pack_bundle
    from regimex.data_catalog import content_hash
    from regimex.model_bundles import ModelBundle, ParentModelEvidence

    greedy, valid, train = fitted()
    assert isinstance(greedy.predictor, AdaptivePredictor)
    config = greedy.predictor.configuration
    multi = AdaptiveBackend().fit(
        train,
        valid,
        parameters=AdaptiveParameters(trees=5, max_depth=4, feature_set="multiscale"),
        policy=config.training_policy,
        analysis=config.analysis,
        runtime=config.training_runtime,
        evaluator=lambda p: float(p[:, 1].mean()),
    )
    assert isinstance(multi.predictor, AdaptivePredictor)
    parents = []
    for name, candidate in (("greedy", greedy), ("multiscale", multi)):
        origin = provenance(candidate).model_copy(update={"model_version_id": name})
        sha = AdaptiveBackend().write_bundle(
            candidate, tmp_path / name, provenance=origin, analysis=config.analysis
        )
        manifest = ModelBundle.model_validate_json((tmp_path / name / "manifest.json").read_bytes())
        parents.append(ParentModelEvidence.from_bundle(manifest, sha))
    greedy_sha, multi_sha = (p.manifest_sha256 for p in parents)
    primary = provenance(multi)
    lineage = BundleProvenance.model_validate(
        {
            **primary.model_dump(),
            "model_version_id": "example-blend",
            "training": {
                **primary.training.model_dump(),
                "parent_models": tuple(parents),
                "validation_dataset_sha256": "a" * 64,
            },
            "environment_sha256": content_hash(blend_environment(greedy_sha, multi_sha, 0.25)),
        }
    )
    sha = pack_bundle(
        tmp_path / "blend",
        greedy_bundle=tmp_path / "greedy",
        greedy_sha256=greedy_sha,
        multiscale_bundle=tmp_path / "multiscale",
        multiscale_sha256=multi_sha,
        fraction=0.25,
        provenance=lineage,
    )
    return sha, valid, greedy.predictor, multi.predictor


def test_blend_restores_exact_mixture_without_parent_directories_and_enforces_combined_budget(
    tmp_path: Path,
) -> None:
    from regimex.backends.adaptive_blend import BlendedAdaptivePredictor

    sha, valid, greedy, multi = blended_bundle(tmp_path)
    (tmp_path / "greedy").rename(tmp_path / "original-greedy")
    (tmp_path / "multiscale").rename(tmp_path / "original-multi")
    budget = greedy.numeric_bytes + multi.numeric_bytes
    runtime = InferenceRuntime(threads=1, model_cache_budget_bytes=budget)
    restored = AdaptiveBackend().load_bundle(
        tmp_path / "blend", manifest_sha256=sha, runtime=runtime
    )
    assert isinstance(restored, BlendedAdaptivePredictor)
    expected = 0.75 * greedy.predict(valid, runtime=runtime) + 0.25 * multi.predict(
        valid, runtime=runtime
    )
    np.testing.assert_array_equal(expected, restored.predict(valid, runtime=runtime))
    assert restored.numeric_bytes == budget
    small = InferenceRuntime(model_cache_budget_bytes=budget - 1)
    with pytest.raises(ResourceBudgetExceeded):
        AdaptiveBackend().load_bundle(tmp_path / "blend", manifest_sha256=sha, runtime=small)
    with pytest.raises(ResourceBudgetExceeded):
        restored.predict(valid, runtime=small)


@pytest.mark.parametrize("change", ["source", "extra", "recipe", "lineage", "truncated"])
def test_blend_checks_nested_sources_and_selection_after_outer_reseal(
    tmp_path: Path, change: str
) -> None:
    import zipfile

    from regimex.storage import digest

    blended_bundle(tmp_path)
    root = tmp_path / "blend"
    if change == "source":
        archive = root / "forests.zip"
        with zipfile.ZipFile(archive) as old:
            payloads = {n: old.read(n) for n in old.namelist()}
        payloads["greedy/weights.npz"] = payloads["multiscale/weights.npz"]
        with zipfile.ZipFile(archive, "w") as rebuilt:
            for key, payload in payloads.items():
                rebuilt.writestr(key, payload)
    elif change == "extra":
        with zipfile.ZipFile(root / "forests.zip", "a") as archive_writer:
            archive_writer.writestr("other-file", "unregistered")
    elif change == "truncated":
        (root / "forests.zip").write_bytes(b"truncated")
    elif change == "recipe":
        write_json(root / "validation.json", {"score": 0.0})
    else:
        manifest = json.loads((root / "manifest.json").read_bytes())
        manifest["training"]["parent_models"].reverse()
        write_json(root / "manifest.json", manifest)
    sha = digest(root / "manifest.json") if change == "lineage" else rehash(root)
    with pytest.raises(ValueError):
        AdaptiveBackend().load_bundle(root, manifest_sha256=sha, runtime=InferenceRuntime())


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


@pytest.mark.parametrize(
    "overrides", [{}, {"feature_set": "multiscale", "sample_weighting": "interval"}]
)
def test_registered_backend_fits_scores_and_restores_a_candidate(
    tmp_path: Path, overrides: dict[str, str]
) -> None:
    data, _ = prepared_data()
    intake = request().model_copy(
        update={
            "architecture_id": "adaptive",
            "parameters": {"trees": 5, "max_depth": 4, **overrides},
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
    assert [row["trees"] for row in events] == [1, 2, 3, 5]
    assert all(row["max_depth"] == 4 and row["seed"] == 35 for row in events)
    assert events[-1]["validation_segment_macro_f1"] == result.validation_segment_macro_f1
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


def test_validation_prefix_scores_use_original_grid_and_full_forest_selection() -> None:
    result, valid, _ = fitted()
    predictor = result.predictor
    assert isinstance(predictor, AdaptivePredictor)
    curves = dict(predictor.validation_predictions(valid, memory_budget_bytes=1024**2))
    assert list(curves) == [1, 2, 3]
    for row, (trees, probabilities) in zip(result.history, curves.items(), strict=True):
        prefix = replace(predictor.model, offsets=predictor.model.offsets[:trees])
        expected = np.concatenate(
            [prefix.predict(features(rows, predictor.analysis)) for rows in valid.observations]
        )
        np.testing.assert_array_equal(probabilities, expected)
        assert row["validation_segment_macro_f1"] == float(expected[:, 0].mean())
    assert result.validation_score == result.history[-1]["validation_segment_macro_f1"]

    feature_bytes = (
        max(i.step_count for i in valid.info)
        * len(features(valid.observations[0], predictor.analysis)[0])
        * 8
        * 4
    )
    one_matrix = curves[3].nbytes
    budget = predictor.numeric_bytes + feature_bytes + one_matrix
    constrained = dict(predictor.validation_predictions(valid, memory_budget_bytes=budget))
    assert list(constrained) == [3]
    np.testing.assert_array_equal(constrained[3], curves[3])
    assert [
        trees
        for trees, _ in predictor.validation_predictions(
            valid, memory_budget_bytes=budget + one_matrix
        )
    ] == [1, 3]
    with pytest.raises(ResourceBudgetExceeded):
        next(predictor.validation_predictions(valid, memory_budget_bytes=1))


def test_restore_forest_that_recorded_only_the_final_validation_score(tmp_path: Path) -> None:
    result, valid, _ = fitted()
    backend = AdaptiveBackend()
    original = replace(result, history=result.history[-1:])
    sha = backend.write_bundle(
        original, tmp_path, provenance=provenance(original), analysis=result.predictor.analysis
    )
    restored = backend.load_bundle(tmp_path, manifest_sha256=sha, runtime=InferenceRuntime())
    np.testing.assert_array_equal(
        restored.predict(valid, runtime=InferenceRuntime()),
        result.predictor.predict(valid, runtime=InferenceRuntime()),
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
    "member", ["history", "preprocessing", "validation", "configuration", "weights", "feature_set"]
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
        path = tmp_path / ("configuration.json" if member == "feature_set" else member + ".json")
        changed = json.loads(path.read_bytes())
        if member == "history":
            changed["fits"][0]["seed"] += 1
        elif member == "validation":
            changed["score"] = 1 - changed["score"]
        elif member == "configuration":
            changed["codec_version"] = "unknown/99"
        elif member == "feature_set":
            changed["parameters"]["feature_set"] = "multiscale"
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
