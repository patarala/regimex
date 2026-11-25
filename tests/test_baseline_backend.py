"""Backend behavior, original-grid alignment, and safe numeric artifact reload."""

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from regimex.backends.baseline import BaselineBackend, BaselinePredictor
from regimex.baseline import LinearModel, aggregate, fit_baseline
from regimex.contracts import Observation
from regimex.directions import AnalysisConfig
from regimex.exports import ExportInstrument, ExportObservation, InstrumentUnits, ModelTimeGrid
from regimex.features import (
    FEATURE_NAMES,
    NUMERIC_NAMES,
    DoubleArray,
    FloatArray,
    Preprocessor,
    compute_features,
)
from regimex.model_bundles import ModelBundle, ReferenceIdentity, TrainingEvidence
from regimex.model_contracts import BundleProvenance, FitResult, ObservationInputs, TrainingInputs
from regimex.model_registry import (
    InferenceRuntime,
    LogisticParameters,
    SummaryMode,
    TrainingPolicy,
    TrainingRuntime,
)
from regimex.prediction_inputs import PredictionSequenceInfo, PreparedPredictionInputs
from regimex.resource_limits import ResourceBudgetExceeded
from regimex.storage import digest, json_bytes, write_json


def inputs(prefix: str, lengths: tuple[int, ...] = (15, 21, 18)) -> ObservationInputs:
    rng = np.random.default_rng(71 if prefix == "train" else 91)
    info = tuple(
        PredictionSequenceInfo(
            experiment_id=f"{prefix}-{index}", instrument_id="A/Q", step_count=length
        )
        for index, length in enumerate(lengths)
    )
    rows = tuple(
        tuple(
            ExportObservation(
                instrument_id="A/Q",
                step=step,
                last_price=round(100 + index * step / 10 + float(rng.normal(scale=0.02)), 6),
                trade_count=1,
                trade_volume=1.0,
                buy_order_count=None if step % 7 == 0 else index + 1,
                buy_order_volume=None if step % 7 == 0 else float(index + 1),
                sell_order_count=1,
                sell_order_volume=1.0,
                best_bid=None,
                best_ask=None,
                price_state="observed",
                bid_state="unknown",
                ask_state="unknown",
            )
            for step in range(length)
        )
        for index, length in enumerate(lengths)
    )
    grids = tuple(
        ModelTimeGrid(
            origin=0.0,
            unit="step",
            bin_width=1.0,
            completed_horizon=float(length),
            step_count=length,
            interval_bounds="[start,stop)",
            terminal_bin="full_bins_only",
        )
        for length in lengths
    )
    instruments = tuple(
        ExportInstrument(
            instrument_id="A/Q",
            base_asset="A",
            quote_asset="Q",
            initial_price=100.0,
            units=InstrumentUnits(
                price_unit="quote_asset_per_base_asset",
                volume_unit="base_asset",
                price_decimal_places=8,
                volume_decimal_places=8,
            ),
        )
        for _ in lengths
    )
    return ObservationInputs(info, rows, grids, instruments)


def prepared(inputs: ObservationInputs, preprocessor: Preprocessor) -> PreparedPredictionInputs:
    return PreparedPredictionInputs(
        preprocessor.transform(
            np.concatenate([compute_features(rows) for rows in inputs.observations])
        ),
        np.r_[0, np.cumsum([info.step_count for info in inputs.info])].astype(np.int64),
        inputs.info,
    )


def fit(
    *, analysis: AnalysisConfig | None = None, summaries: SummaryMode = SummaryMode.LOCAL_AND_GLOBAL
) -> tuple[BaselineBackend, FitResult, ObservationInputs]:
    backend = BaselineBackend()
    training, validation = inputs("train"), inputs("validation")
    targets = np.repeat(np.arange(3), (15, 21, 18)).astype(np.int64)
    targets[::11] = -1
    result = backend.fit(
        TrainingInputs(training, targets),
        validation,
        parameters=LogisticParameters(
            regularization=(0.1, 1.0), context_sizes=(4, 16), summaries=summaries
        ),
        policy=TrainingPolicy(optimizer="lbfgs", seeds=(11,)),
        analysis=analysis or AnalysisConfig(min_duration=4),
        runtime=TrainingRuntime(threads=1),
        evaluator=lambda probabilities: float(probabilities[:, 0].mean()),
    )
    return backend, result, validation


def provenance(result: FitResult) -> BundleProvenance:
    reference = ReferenceIdentity(
        reference_set_id="fixture-reference",
        kind="automatic",
        policy="fixture/1.0",
        content_sha256="1" * 64,
        direction_semantics_sha256="2" * 64,
    )
    return BundleProvenance(
        model_version_id="fixture-baseline",
        analysis_profile="fixture-profile",
        analysis_profile_sha256=hashlib.sha256(
            json_bytes(result.predictor.analysis.model_dump(mode="json"))
        ).hexdigest(),
        training=TrainingEvidence(
            run_id="fixture-run",
            dataset_sha256="3" * 64,
            split_manifest_sha256="4" * 64,
            labels=reference,
            validation_references=reference,
            seeds=(11,),
            selected_seed=11,
            selection_metric="decoded_segment_macro_f1/iou-0.50",
            validation_segment_macro_f1=result.validation_score,
        ),
        build_id="fixture-build",
        environment_sha256=hashlib.sha256(json_bytes(result.environment)).hexdigest(),
    )


def bundle(
    tmp_path: Path, *, analysis: AnalysisConfig | None = None
) -> tuple[BaselineBackend, FitResult, ObservationInputs, str]:
    backend, result, validation = fit(analysis=analysis)
    sha = backend.write_bundle(
        result,
        tmp_path / "bundle",
        provenance=provenance(result),
        analysis=result.predictor.analysis,
    )
    return backend, result, validation, sha


def rehash(root: Path) -> str:
    """Keep envelope checks valid so corruption tests exercise the backend codec."""
    manifest = ModelBundle.model_validate_json((root / "manifest.json").read_bytes())
    updated = manifest.model_copy(
        update={
            "members": tuple(
                member.model_copy(
                    update={
                        "size_bytes": (root / member.path).stat().st_size,
                        "sha256": digest(root / member.path),
                    }
                )
                for member in manifest.members
            ),
        }
    )
    write_json(root / "manifest.json", updated.model_dump(mode="json"))
    return digest(root / "manifest.json")


def test_backend_matches_numeric_fit_and_training_only_preprocessing() -> None:
    _, result, validation = fit()
    fitted = cast(BaselinePredictor, result.predictor)
    training = inputs("train")
    targets = np.repeat(np.arange(3), (15, 21, 18)).astype(np.int64)
    targets[::11] = -1
    preprocessing = Preprocessor.fit(
        np.concatenate([compute_features(rows) for rows in training.observations])
    )
    prepared_train = prepared(training, preprocessing)
    prepared_validation = prepared(validation, preprocessing)
    expected, history, selected = fit_baseline(
        prepared_train,
        targets,
        prepared_validation,
        regularization_candidates=(0.1, 1.0),
        seed=11,
        threads=1,
        context_sizes=(4, 16),
        scorer=lambda probabilities: float(probabilities[:, 0].mean()),
    )
    np.testing.assert_array_equal(fitted.preprocessor.mean, preprocessing.mean)
    np.testing.assert_array_equal(fitted.model.coefficients, expected.coefficients)
    assert result.history == tuple(history)
    assert result.selected_settings == {"regularization": selected}
    for batch_limit in (1, 2, 4096):
        actual = fitted.predict(
            validation, runtime=InferenceRuntime(threads=1, max_batch_size=batch_limit)
        )
        # BLAS reductions may round differently for different prediction batch sizes.
        np.testing.assert_allclose(
            actual, expected.predict(prepared_validation), rtol=1e-13, atol=1e-15
        )
    altered = inputs("validation", (9, 12, 7))
    prior = fitted.preprocessor.mean.copy()
    fitted.predict(altered, runtime=InferenceRuntime(threads=1))
    np.testing.assert_array_equal(fitted.preprocessor.mean, prior)


def test_inference_extracts_only_the_current_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    _, result, validation = fit()
    events: list[tuple[str, int]] = []
    predict = LinearModel.predict

    def extract(rows: tuple[Observation, ...]) -> FloatArray:
        events.append(("extract", len(rows)))
        return compute_features(rows)

    def score(model: LinearModel, data: PreparedPredictionInputs) -> DoubleArray:
        events.append(("predict", len(data.values)))
        return predict(model, data)

    monkeypatch.setattr("regimex.backends.baseline.compute_features", extract)
    monkeypatch.setattr(LinearModel, "predict", score)
    result.predictor.predict(validation, runtime=InferenceRuntime(threads=1, max_batch_size=2))
    assert events == [
        ("extract", 15),
        ("extract", 21),
        ("predict", 36),
        ("extract", 18),
        ("predict", 18),
    ]


def test_validation_obeys_the_training_blas_limit() -> None:
    from threadpoolctl import threadpool_info, threadpool_limits  # type: ignore[import-untyped]

    training, validation = inputs("train"), inputs("validation")

    def evaluate(_: DoubleArray) -> float:
        assert all(
            pool["num_threads"] == 1 for pool in threadpool_info() if pool["user_api"] == "blas"
        )
        return 0.5

    with threadpool_limits(limits=2):
        BaselineBackend().fit(
            TrainingInputs(training, np.repeat(np.arange(3), (15, 21, 18)).astype(np.int64)),
            validation,
            parameters=LogisticParameters(regularization=(1.0,), context_sizes=(4,)),
            policy=TrainingPolicy(optimizer="lbfgs", seeds=(11,)),
            analysis=AnalysisConfig(),
            runtime=TrainingRuntime(threads=1),
            evaluator=evaluate,
        )


def test_local_summaries_preserve_distant_features_when_a_price_is_missing() -> None:
    original = inputs("validation", (24,))
    rows = original.observations[0]
    gap = tuple(
        row.model_copy(update={"last_price": None, "price_state": "unknown"})
        if 12 <= row.step < 16
        else row
        for row in rows
    )
    masked = replace(original, observations=(gap,))
    raw = np.concatenate([compute_features(seq) for seq in original.observations])
    prep = Preprocessor.fit(raw)
    before, after = prepared(original, prep), prepared(masked, prep)
    for mode in (SummaryMode.LOCAL, SummaryMode.LOCAL_OBSERVED):
        np.testing.assert_array_equal(
            aggregate(before, (4,), mode)[:10], aggregate(after, (4,), mode)[:10]
        )
    assert not np.array_equal(aggregate(before, (4,))[0], aggregate(after, (4,))[0])


def test_observed_statistics_exclude_fill_and_preserve_missing_indicators() -> None:
    observations = inputs("validation", (3,))
    width = len(FEATURE_NAMES)
    values = np.zeros((3, width), dtype=np.float32)
    values[:, 0] = [1.0, -99.0, 5.0]
    values[:, len(NUMERIC_NAMES)] = [0.0, 1.0, 0.0]
    values[:, 1] = -99.0
    values[:, len(NUMERIC_NAMES) + 1] = 1.0
    prepared = PreparedPredictionInputs(values, np.array([0, 3]), observations.info)
    summaries = aggregate(prepared, (3,), SummaryMode.LOCAL_OBSERVED)
    window = summaries[1].reshape(4, width)
    np.testing.assert_allclose(window[:, 0], [3, 2, 1, 5])
    np.testing.assert_array_equal(window[:, 1], np.zeros(4))
    np.testing.assert_allclose(window[:, len(NUMERIC_NAMES)], [1 / 3, np.sqrt(2) / 3, 0, 1])
    np.testing.assert_array_equal(window[:, len(NUMERIC_NAMES) + 1], [1, 0, 1, 1])


@pytest.mark.parametrize("mode", list(SummaryMode))
def test_summary_mode_survives_reload(tmp_path: Path, mode: SummaryMode) -> None:
    backend, fitted, validation = fit(summaries=mode)
    root = tmp_path / "bundle"
    runtime = InferenceRuntime(threads=1)
    sha = backend.write_bundle(
        fitted, root, provenance=provenance(fitted), analysis=fitted.predictor.analysis
    )
    saved = json.loads((root / "configuration.json").read_bytes())
    assert saved["codec_version"] == "aggregate-logistic/3.0"
    assert saved["parameters"]["summaries"] == mode
    restored = backend.load_bundle(root, manifest_sha256=sha, runtime=runtime)
    assert restored.model.summaries == mode
    assert restored.model.accumulation_dtype is np.float64
    np.testing.assert_array_equal(
        restored.predict(validation, runtime=runtime),
        fitted.predictor.predict(validation, runtime=runtime),
    )
    if mode != SummaryMode.LOCAL_AND_GLOBAL:
        saved["codec_version"] = "aggregate-logistic/1.0"
        write_json(root / "configuration.json", saved)
        with pytest.raises(ValueError, match="summary mode differs from its codec"):
            backend.load_bundle(root, manifest_sha256=rehash(root), runtime=runtime)


@pytest.mark.parametrize(
    ("mode", "codec"),
    [
        (SummaryMode.LOCAL_AND_GLOBAL, "aggregate-logistic/1.0"),
        (SummaryMode.LOCAL, "aggregate-logistic/2.0"),
    ],
)
def test_saved_weights_retain_their_window_arithmetic(
    tmp_path: Path, mode: SummaryMode, codec: str
) -> None:
    backend, fitted, _ = fit(summaries=mode)
    root = tmp_path / "bundle"
    backend.write_bundle(
        fitted, root, provenance=provenance(fitted), analysis=fitted.predictor.analysis
    )
    config = json.loads((root / "configuration.json").read_bytes())
    if mode == SummaryMode.LOCAL_AND_GLOBAL:
        del config["parameters"]["summaries"]
    config["codec_version"] = codec
    write_json(root / "configuration.json", config)

    # Make the growth logit read the trade volume deviation in the 16-step window.
    with np.load(root / "weights.npz") as weights:
        state = dict(weights)
    state["coefficients"].fill(0)
    volume = FEATURE_NAMES.index("log_trade_volume")
    state["coefficients"][1, 5 * len(FEATURE_NAMES) + volume] = 1e7
    state["mean"].fill(0)
    state["scale"].fill(1)
    state["intercept"].fill(0)
    np.savez(root / "weights.npz", **state)

    data = inputs("validation", (32,))
    rows = tuple(row.model_copy(update={"trade_volume": 4.0}) for row in data.observations[0])
    data = replace(data, observations=(rows,))
    runtime = InferenceRuntime(threads=1)
    predictor = backend.load_bundle(root, manifest_sha256=rehash(root), runtime=runtime)
    values = prepared(data, predictor.preprocessor).values[:16]
    old_deviation = values.std(axis=0)[volume]
    assert old_deviation > 0
    odds = np.exp(float(old_deviation) * 1e7)
    assert predictor.predict(data, runtime=runtime)[16, 1] == pytest.approx(odds / (odds + 2))

    config["codec_version"] = "aggregate-logistic/3.0"
    write_json(root / "configuration.json", config)
    predictor = backend.load_bundle(root, manifest_sha256=rehash(root), runtime=runtime)
    assert predictor.predict(data, runtime=runtime)[16, 1] == pytest.approx(1 / 3)


def test_empty_contexts_require_whole_sequence_statistics() -> None:
    assert LogisticParameters(context_sizes=()).summaries.whole_sequence
    with pytest.raises(ValueError, match="local-only summaries"):
        LogisticParameters(context_sizes=(), summaries=SummaryMode.LOCAL_OBSERVED)


def test_original_grid_and_no_cross_sequence_context() -> None:
    _, result, validation = fit()
    full = result.predictor.predict(validation, runtime=InferenceRuntime(threads=1))
    separate = []
    for index, info in enumerate(validation.info):
        one = ObservationInputs(
            (info,),
            (validation.observations[index],),
            (validation.time_grids[index],),
            (validation.instruments[index],),
        )
        separate.append(result.predictor.predict(one, runtime=InferenceRuntime(threads=1)))
    assert full.shape == (sum(info.step_count for info in validation.info), 3)
    np.testing.assert_allclose(full, np.concatenate(separate), rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(full.sum(axis=1), 1, atol=1e-12)


@pytest.mark.parametrize(
    "analysis", [AnalysisConfig(version="2.0", min_duration=4), AnalysisConfig(min_duration=4)]
)
def test_fresh_process_reload_has_no_torch_sklearn_or_fitting(
    tmp_path: Path, analysis: AnalysisConfig
) -> None:
    backend, result, validation, sha = bundle(tmp_path, analysis=analysis)
    restored = backend.load_bundle(
        tmp_path / "bundle", manifest_sha256=sha, runtime=InferenceRuntime()
    )
    assert restored.analysis == analysis
    expected = result.predictor.predict(validation, runtime=InferenceRuntime(threads=1))
    np.testing.assert_array_equal(
        restored.predict(validation, runtime=InferenceRuntime(threads=1)), expected
    )
    write_json(
        tmp_path / "observations.json",
        {
            "info": [item.model_dump(mode="json") for item in validation.info],
            "observations": [
                [row.model_dump(mode="json") for row in rows] for rows in validation.observations
            ],
            "time_grids": [item.model_dump(mode="json") for item in validation.time_grids],
            "instruments": [item.model_dump(mode="json") for item in validation.instruments],
        },
    )
    script = """
import importlib.abc
import json
import sys
from pathlib import Path
class BlockScientificTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'sklearn'}:
            raise AssertionError('inference imported a training runtime: ' + fullname)
sys.meta_path.insert(0, BlockScientificTraining())
import numpy as np
from regimex.backends.baseline import BaselineBackend
from regimex.features import Preprocessor
from regimex.model_registry import InferenceRuntime
from regimex.model_contracts import ObservationInputs
from regimex.prediction_inputs import PredictionSequenceInfo
from regimex.exports import ExportObservation, ModelTimeGrid, ExportInstrument
def forbidden(*args, **kwargs):
    raise AssertionError('inference attempted fitting')
Preprocessor.fit = forbidden
root = Path(sys.argv[1])
data = json.loads((root / 'observations.json').read_text())
inputs = ObservationInputs(
    tuple(PredictionSequenceInfo.model_validate(item) for item in data['info']),
    tuple(
        tuple(ExportObservation.model_validate(item) for item in rows)
        for rows in data['observations']
    ),
    tuple(ModelTimeGrid.model_validate(item) for item in data['time_grids']),
    tuple(ExportInstrument.model_validate(item) for item in data['instruments']),
)
predictor = BaselineBackend().load_bundle(
    root / 'bundle', manifest_sha256=sys.argv[2], runtime=InferenceRuntime()
)
np.save(
    root / 'probabilities.npy', predictor.predict(inputs, runtime=InferenceRuntime(threads=1)),
    allow_pickle=False
)
assert not any(name.split('.')[0] in {'torch', 'sklearn'} for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), sha],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    np.testing.assert_array_equal(
        np.load(tmp_path / "probabilities.npy", allow_pickle=False), expected
    )


@pytest.mark.parametrize(
    "corruption", ["dtype", "shape", "nan", "scale", "masks", "object", "extra"]
)
def test_codec_rejects_invalid_numeric_state_even_with_valid_checksums(
    tmp_path: Path, corruption: str
) -> None:
    backend, _, _, _ = bundle(tmp_path)
    root = tmp_path / "bundle"
    with np.load(root / "preprocessing.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if corruption == "dtype":
        arrays["mean"] = arrays["mean"].astype(np.float32)
    elif corruption == "shape":
        arrays["mean"] = arrays["mean"][:-1]
    elif corruption == "nan":
        arrays["mean"][0] = np.nan
    elif corruption == "scale":
        arrays["scale"][0] = 0
    elif corruption == "masks":
        arrays["mean"][len(NUMERIC_NAMES)] = 1
    elif corruption == "object":
        arrays["mean"] = np.array(["unsafe"] * len(FEATURE_NAMES), dtype=object)
    else:
        arrays["unknown"] = np.ones(1, dtype=np.float64)
    np.savez(root / "preprocessing.npz", **arrays)
    sha = rehash(root)
    with pytest.raises(ValueError, match=r"baseline|preprocessing"):
        backend.load_bundle(root, manifest_sha256=sha, runtime=InferenceRuntime())


@pytest.mark.parametrize(
    "corruption", ["context", "feature", "solver", "policy", "history", "environment", "profile"]
)
def test_configuration_and_provenance_remain_bound(tmp_path: Path, corruption: str) -> None:
    backend, _, _, _ = bundle(tmp_path)
    root = tmp_path / "bundle"
    configuration = json.loads((root / "configuration.json").read_text())
    if corruption == "context":
        configuration["parameters"]["context_sizes"] = [8, 16]
    elif corruption == "feature":
        configuration["feature_names"][0] = "target"
    elif corruption == "solver":
        configuration["solver"] = "adamw"
    elif corruption == "policy":
        configuration["training_policy"]["epochs"] = 99
    elif corruption == "history":
        history = json.loads((root / "history.json").read_text())
        history["candidates"][0]["validation_segment_macro_f1"] = 1
        write_json(root / "history.json", history)
    elif corruption == "environment":
        configuration["environment"]["python"] = "unverified"
    else:
        configuration["analysis"]["min_duration"] = 8
    write_json(root / "configuration.json", configuration)
    sha = rehash(root)
    with pytest.raises(ValueError):
        backend.load_bundle(root, manifest_sha256=sha, runtime=InferenceRuntime())


def test_provenance_cannot_rebind_fitted_selection_or_analysis(tmp_path: Path) -> None:
    backend, result, _ = fit()
    supplied = provenance(result)
    for bad in (
        supplied.model_copy(update={"environment_sha256": "f" * 64}),
        supplied.model_copy(
            update={
                "training": supplied.training.model_copy(
                    update={"validation_segment_macro_f1": 1.0}
                )
            }
        ),
    ):
        with pytest.raises(ValueError, match="provenance"):
            backend.write_bundle(
                result, tmp_path / "bundle", provenance=bad, analysis=result.predictor.analysis
            )
    with pytest.raises(ValueError, match="analysis profile"):
        backend.write_bundle(
            result,
            tmp_path / "bundle",
            provenance=supplied,
            analysis=AnalysisConfig(min_duration=8),
        )
    with pytest.raises(ValueError, match="history"):
        backend.write_bundle(
            replace(result, selected_settings={"regularization": 99.0}),
            tmp_path / "bundle",
            provenance=supplied,
            analysis=result.predictor.analysis,
        )
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    "mode",
    [
        "classes",
        "seeds",
        "optimizer",
        "irrelevant",
        "overlap",
        "memory",
        "score",
        "wrong-parameters",
    ],
)
def test_invalid_training_fails_explicitly(mode: str) -> None:
    training, validation = inputs("train"), inputs("validation")
    targets = np.repeat(np.arange(3), (15, 21, 18)).astype(np.int64)
    policy = TrainingPolicy(optimizer="lbfgs", seeds=(11,))
    runtime = TrainingRuntime(threads=1)
    parameters = LogisticParameters(regularization=(1.0,), context_sizes=(4,))
    if mode == "classes":
        targets[:] = 0
    elif mode == "seeds":
        policy = policy.model_copy(update={"seeds": (11, 23)})
    elif mode == "optimizer":
        policy = policy.model_copy(update={"optimizer": "adamw"})
    elif mode == "irrelevant":
        policy = policy.model_copy(update={"learning_rate": 0.5})
    elif mode == "overlap":
        validation = training
    elif mode == "memory":
        runtime = runtime.model_copy(update={"memory_budget_bytes": 1})
    elif mode == "wrong-parameters":
        with pytest.raises(TypeError, match="LogisticParameters"):
            BaselineBackend().fit(
                TrainingInputs(training, targets),
                validation,
                parameters=AnalysisConfig(),
                policy=policy,
                analysis=AnalysisConfig(),
                runtime=runtime,
                evaluator=lambda _: 0.5,
            )
        return
    error = ResourceBudgetExceeded if mode == "memory" else ValueError
    with pytest.raises(error):
        BaselineBackend().fit(
            TrainingInputs(training, targets),
            validation,
            parameters=parameters,
            policy=policy,
            analysis=AnalysisConfig(),
            runtime=runtime,
            evaluator=lambda _: float("nan") if mode == "score" else 0.5,
        )


def test_training_time_limit_is_a_resource_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    training, validation = inputs("train"), inputs("validation")
    clock = iter((0.0, 2.0))
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        "regimex.backends.baseline.compute_features",
        lambda _: pytest.fail("feature extraction continued after the deadline"),
    )
    with pytest.raises(ResourceBudgetExceeded, match="time budget"):
        BaselineBackend().fit(
            TrainingInputs(training, np.repeat(np.arange(3), (15, 21, 18)).astype(np.int64)),
            validation,
            parameters=LogisticParameters(regularization=(1.0,), context_sizes=(4,)),
            policy=TrainingPolicy(optimizer="lbfgs", seeds=(11,)),
            analysis=AnalysisConfig(),
            runtime=TrainingRuntime(threads=1, time_budget_seconds=1),
            evaluator=lambda _: 0.5,
        )


def test_model_cache_budget_is_checked_before_restore_or_predict(tmp_path: Path) -> None:
    backend, result, validation, sha = bundle(tmp_path)
    runtime = InferenceRuntime(model_cache_budget_bytes=1)
    with pytest.raises(ResourceBudgetExceeded, match="cache budget"):
        backend.load_bundle(tmp_path / "bundle", manifest_sha256=sha, runtime=runtime)
    with pytest.raises(ResourceBudgetExceeded, match="cache budget"):
        result.predictor.predict(validation, runtime=runtime)


def test_observation_boundary_rejects_training_metadata_and_grid_damage() -> None:
    original = inputs("fixture")
    with pytest.raises(ValueError, match="contiguous timestamp"):
        replace(
            original,
            observations=(
                (
                    original.observations[0][0].model_copy(update={"step": 1}),
                    *original.observations[0][1:],
                ),
                *original.observations[1:],
            ),
        )
    with pytest.raises(ValueError, match="undeclared fields"):
        replace(
            original,
            info=(original.info[0].model_copy(update={"seed": 11}), *original.info[1:]),
        )
    with pytest.raises(ValueError, match="original time grids"):
        replace(
            original,
            time_grids=(
                original.time_grids[0].model_copy(
                    update={"step_count": 14, "completed_horizon": 14.0}
                ),
                *original.time_grids[1:],
            ),
        )


def test_training_targets_are_a_snapshot_and_never_an_inference_container() -> None:
    _, result, raw = fit()
    targets = np.zeros(raw.row_count, dtype=np.int64)
    training = TrainingInputs(raw, targets)
    targets[:] = 2
    np.testing.assert_array_equal(training.targets, np.zeros(raw.row_count, dtype=np.int64))
    with pytest.raises(TypeError, match="observation-only"):
        result.predictor.predict(cast(ObservationInputs, training), runtime=InferenceRuntime())


@pytest.mark.parametrize("score", [-1.0, 1.1, float("inf")])
def test_no_candidate_can_return_an_invalid_validation_score(score: float) -> None:
    training, validation = inputs("train"), inputs("validation")
    with pytest.raises(ValueError, match="validation score"):
        BaselineBackend().fit(
            TrainingInputs(training, np.repeat(np.arange(3), (15, 21, 18)).astype(np.int64)),
            validation,
            parameters=LogisticParameters(regularization=(1.0,), context_sizes=(4,)),
            policy=TrainingPolicy(optimizer="lbfgs", seeds=(11,)),
            analysis=AnalysisConfig(),
            runtime=TrainingRuntime(threads=1),
            evaluator=lambda _: score,
        )
