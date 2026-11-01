"""Real TCN fitting, raw-input isolation and restricted artifact restoration."""

import hashlib
import json
import pickle
import subprocess
import sys
import time
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from torch import nn

from regimex.backends.tcn import TCNBackend, TCNPredictor
from regimex.directions import AnalysisConfig
from regimex.exports import ExportInstrument, ExportObservation, InstrumentUnits, ModelTimeGrid
from regimex.features import compute_features
from regimex.model_bundles import ModelBundle, ReferenceIdentity, TrainingEvidence
from regimex.model_contracts import BundleProvenance, FitResult, ObservationInputs, TrainingInputs
from regimex.model_registry import InferenceRuntime, TCNParameters, TrainingPolicy, TrainingRuntime
from regimex.prediction_inputs import (
    PredictionInputs,
    PredictionSequenceInfo,
    PreparedPredictionInputs,
)
from regimex.resource_limits import ResourceBudgetExceeded
from regimex.storage import digest, json_bytes, write_json
from regimex.temporal import (
    TemporalPreprocessor,
    fit_tcn,
    predict_tcn,
    temporal_features,
)


def inputs(prefix: str, lengths: tuple[int, ...] = (24, 32, 28)) -> ObservationInputs:
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
                last_price=None
                if step % 13 == 0
                else float(100 * np.exp((0, 0.002, -0.002)[index % 3] * step)),
                trade_count=None,
                trade_volume=None,
                buy_order_count=None,
                buy_order_volume=None,
                sell_order_count=0,
                sell_order_volume=0.0,
                best_bid=None,
                best_ask=None,
                price_state="unknown" if step % 13 == 0 else "observed",
                bid_state="unknown",
                ask_state="unknown",
            )
            for step in range(length)
        )
        for index, length in enumerate(lengths)
    )
    grids = tuple(
        ModelTimeGrid(
            origin=4.0,
            unit="model-second",
            bin_width=2.0,
            completed_horizon=4.0 + 2 * length,
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


def raw(inputs: ObservationInputs) -> PredictionInputs:
    return PredictionInputs(
        np.concatenate([compute_features(rows) for rows in inputs.observations]),
        np.r_[0, np.cumsum([info.step_count for info in inputs.info])].astype(np.int64),
        inputs.info,
    )


def policy() -> TrainingPolicy:
    return TrainingPolicy(
        optimizer="adamw", seeds=(11, 23), epochs=3, batch_size=3, validation_interval=1
    )


def fit() -> tuple[TCNBackend, FitResult, ObservationInputs]:
    backend = TCNBackend()
    training, validation = inputs("train"), inputs("validation")
    targets = np.repeat(np.arange(3), (24, 32, 28)).astype(np.int64)
    targets[::11] = -1
    fitted = backend.fit(
        TrainingInputs(training, targets),
        validation,
        parameters=TCNParameters(channels=4),
        policy=policy(),
        analysis=AnalysisConfig(min_duration=4),
        runtime=TrainingRuntime(threads=1),
        evaluator=lambda probabilities: float(probabilities[:, 0].mean()),
    )
    return backend, fitted, validation


def provenance(result: FitResult) -> BundleProvenance:
    reference = ReferenceIdentity(
        reference_set_id="fixture-reference",
        kind="automatic",
        policy="fixture/1.0",
        content_sha256="1" * 64,
        direction_semantics_sha256="2" * 64,
    )
    return BundleProvenance(
        model_version_id="fixture-tcn",
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
            seeds=policy().seeds,
            selected_seed=result.selected_seed,
            selection_metric="decoded_segment_macro_f1/iou-0.50",
            validation_segment_macro_f1=result.validation_score,
        ),
        build_id="fixture-build",
        environment_sha256=hashlib.sha256(json_bytes(result.environment)).hexdigest(),
    )


def bundle(tmp_path: Path) -> tuple[TCNBackend, FitResult, ObservationInputs, str]:
    backend, result, validation = fit()
    sha = backend.write_bundle(
        result,
        tmp_path / "bundle",
        provenance=provenance(result),
        analysis=result.predictor.analysis,
    )
    return backend, result, validation, sha


def rehash(root: Path) -> str:
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


def test_backend_matches_scientific_fit_and_training_only_preprocessing() -> None:
    _, result, validation = fit()
    predictor = cast(TCNPredictor, result.predictor)
    training = inputs("train")
    numeric = raw(training)
    values = temporal_features(numeric)
    preprocessing = TemporalPreprocessor.fit(values)
    targets = np.repeat(np.arange(3), (24, 32, 28)).astype(np.int64)
    targets[::11] = -1

    def prepared(inputs: ObservationInputs) -> PreparedPredictionInputs:
        numeric = raw(inputs)
        return PreparedPredictionInputs(
            preprocessing.transform(temporal_features(numeric)),
            numeric.offsets,
            inputs.info,
        )

    expected, history, selected_seed = fit_tcn(
        prepared(training),
        targets,
        prepared(validation),
        parameters=TCNParameters(channels=4),
        policy=policy(),
        runtime=TrainingRuntime(threads=1),
        scorer=lambda probabilities: float(probabilities[:, 0].mean()),
    )
    np.testing.assert_array_equal(predictor.preprocessor.mean, preprocessing.mean)
    for name, tensor in expected.state_dict().items():
        torch.testing.assert_close(predictor.model.state_dict()[name], tensor, rtol=0, atol=0)
    np.testing.assert_array_equal(
        predictor.predict(validation, runtime=InferenceRuntime(threads=1)),
        predict_tcn(expected, prepared(validation)),
    )
    assert result.history == tuple(history) and result.selected_seed == selected_seed
    old_mean = predictor.preprocessor.mean.copy()
    predictor.predict(inputs("other", (8, 12, 7)), runtime=InferenceRuntime(threads=1))
    np.testing.assert_array_equal(predictor.preprocessor.mean, old_mean)


def test_predictions_preserve_grid_and_ignore_activity_and_other_instruments() -> None:
    _, result, validation = fit()
    expected = result.predictor.predict(validation, runtime=InferenceRuntime(threads=1))
    altered = replace(
        validation,
        observations=tuple(
            tuple(
                row.model_copy(update={"trade_count": 20, "buy_order_volume": 123.0})
                for row in rows
            )
            for rows in validation.observations
        ),
    )
    np.testing.assert_array_equal(
        result.predictor.predict(altered, runtime=InferenceRuntime(threads=1)), expected
    )
    separate = [
        result.predictor.predict(
            ObservationInputs(
                (info,),
                (validation.observations[index],),
                (validation.time_grids[index],),
                (validation.instruments[index],),
            ),
            runtime=InferenceRuntime(threads=1, max_batch_size=1),
        )
        for index, info in enumerate(validation.info)
    ]
    assert expected.dtype == np.float64 and expected.shape == (validation.row_count, 3)
    np.testing.assert_allclose(np.concatenate(separate), expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(expected.sum(axis=1), 1, rtol=1e-12, atol=1e-12)


def test_restricted_reload_is_exact_and_uses_independent_inference_runtime(tmp_path: Path) -> None:
    backend, fitted, validation, sha = bundle(tmp_path)
    training_threads = cast(TCNPredictor, fitted.predictor).configuration.training_runtime.threads
    runtime = InferenceRuntime(threads=2, max_batch_size=2)
    expected = fitted.predictor.predict(validation, runtime=runtime)
    torch.manual_seed(99)
    prior_rng = torch.random.get_rng_state().clone()
    restored = backend.load_bundle(tmp_path / "bundle", manifest_sha256=sha, runtime=runtime)
    configuration = restored.configuration
    assert restored.implementation_version == "0.2.0"
    assert configuration.codec_version == "temporal-tcn/2.0"
    assert configuration.loss_version == "direction-cross-entropy/2.0"
    assert configuration.smoothing_scope == "equal_adjacent_targets"
    assert configuration.boundary_weight_radius == 4
    assert configuration.boundary_weight_multiplier == 2
    assert torch.get_num_threads() == 2 and training_threads == 1
    assert torch.equal(torch.random.get_rng_state(), prior_rng)
    np.testing.assert_array_equal(restored.predict(validation, runtime=runtime), expected)


@pytest.mark.parametrize(
    ("field", "incompatible"),
    [
        ("loss_version", None),
        ("smoothing_scope", None),
        ("boundary_weight_radius", None),
        ("boundary_weight_multiplier", None),
        ("codec_version", "temporal-tcn/1.0"),
        ("loss_version", "direction-cross-entropy/1.0"),
        ("smoothing_scope", "all_adjacent_targets"),
        ("boundary_weight_radius", 3),
        ("boundary_weight_multiplier", 3),
    ],
)
def test_rejects_missing_or_incompatible_training_loss_metadata(
    tmp_path: Path, field: str, incompatible: str | int | None
) -> None:
    backend, _, _, _ = bundle(tmp_path)
    root = tmp_path / "bundle"
    path = root / "configuration.json"
    configuration = json.loads(path.read_bytes())
    if incompatible is None:
        del configuration[field]
    else:
        configuration[field] = incompatible
    write_json(path, configuration)
    with pytest.raises(ValueError, match="TCN"):
        backend.load_bundle(root, manifest_sha256=rehash(root), runtime=InferenceRuntime(threads=1))


def test_fresh_process_reload_needs_no_sklearn_or_fitting(tmp_path: Path) -> None:
    _, fitted, validation, sha = bundle(tmp_path)
    expected = fitted.predictor.predict(validation, runtime=InferenceRuntime(threads=1))
    np.save(tmp_path / "expected.npy", expected)
    write_json(
        tmp_path / "observations.json",
        {
            "info": [info.model_dump(mode="json") for info in validation.info],
            "observations": [
                [row.model_dump(mode="json") for row in rows] for rows in validation.observations
            ],
            "time_grids": [grid.model_dump(mode="json") for grid in validation.time_grids],
            "instruments": [
                instrument.model_dump(mode="json") for instrument in validation.instruments
            ],
        },
    )
    script = """
import builtins
import json
import sys
from pathlib import Path
original_import = builtins.__import__
def no_sklearn(name, *args, **kwargs):
    if name.split('.')[0] == 'sklearn':
        raise AssertionError('inference imported scikit-learn')
    return original_import(name, *args, **kwargs)
builtins.__import__ = no_sklearn
import numpy as np
from regimex.backends.tcn import TCNBackend
from regimex.model_contracts import ObservationInputs
from regimex.model_registry import InferenceRuntime
from regimex.temporal import TemporalPreprocessor
from regimex.prediction_inputs import PredictionSequenceInfo
from regimex.exports import ExportObservation, ModelTimeGrid, ExportInstrument
def forbidden(*args, **kwargs):
    raise AssertionError('restoration or prediction attempted fitting')
TemporalPreprocessor.fit = forbidden
root = Path(sys.argv[1])
value = json.loads((root / 'observations.json').read_text())
inputs = ObservationInputs(
    tuple(PredictionSequenceInfo.model_validate_json(json.dumps(x)) for x in value['info']),
    tuple(
        tuple(ExportObservation.model_validate_json(json.dumps(x)) for x in rows)
        for rows in value['observations']
    ),
    tuple(ModelTimeGrid.model_validate_json(json.dumps(x)) for x in value['time_grids']),
    tuple(ExportInstrument.model_validate_json(json.dumps(x)) for x in value['instruments']),
)
runtime = InferenceRuntime(threads=1)
predictor = TCNBackend().load_bundle(root / 'bundle', manifest_sha256=sys.argv[2], runtime=runtime)
np.testing.assert_array_equal(
    predictor.predict(inputs, runtime=runtime), np.load(root / 'expected.npy')
)
assert not any(name.split('.')[0] == 'sklearn' for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), sha],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("change", ["dtype", "shape", "nan", "missing", "extra"])
def test_rejects_incompatible_weight_state(tmp_path: Path, change: str) -> None:
    backend, _, _, _ = bundle(tmp_path)
    root = tmp_path / "bundle"
    state = torch.load(root / "weights.pt", weights_only=True, map_location="cpu")
    first = next(iter(state))
    if change == "dtype":
        state[first] = state[first].to(torch.float64)
    elif change == "shape":
        state[first] = state[first].reshape(-1)
    elif change == "nan":
        state[first].reshape(-1)[0] = float("nan")
    elif change == "missing":
        state.pop(first)
    else:
        state["unexpected"] = torch.zeros(1)
    torch.save(state, root / "weights.pt")
    sha = rehash(root)
    with pytest.raises(ValueError, match=r"TCN|restricted"):
        backend.load_bundle(root, manifest_sha256=sha, runtime=InferenceRuntime(threads=1))


@pytest.mark.parametrize("change", ["scale", "mask", "dtype", "shape", "extra"])
def test_rejects_incompatible_preprocessing(tmp_path: Path, change: str) -> None:
    backend, _, _, _ = bundle(tmp_path)
    root = tmp_path / "bundle"
    with np.load(root / "preprocessing.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if change == "scale":
        arrays["scale"][0] = 0
    elif change == "mask":
        arrays["mean"][-1] = 1
    elif change == "dtype":
        arrays["mean"] = arrays["mean"].astype(np.float32)
    elif change == "shape":
        arrays["mean"] = arrays["mean"][:-1]
    else:
        arrays["extra"] = np.zeros(1)
    np.savez(root / "preprocessing.npz", **arrays)
    with pytest.raises(ValueError, match="TCN"):
        backend.load_bundle(root, manifest_sha256=rehash(root), runtime=InferenceRuntime(threads=1))


@pytest.mark.parametrize("member", ["configuration", "validation", "history"])
def test_rejects_cross_field_codec_provenance(tmp_path: Path, member: str) -> None:
    backend, _, _, _ = bundle(tmp_path)
    root = tmp_path / "bundle"
    path = root / f"{member}.json"
    data = json.loads(path.read_bytes())
    if member == "configuration":
        data["parameters"]["dilations"] = [1, 2]
    elif member == "validation":
        data["selected_epoch"] = 1000
    else:
        data["epochs"][0]["seed"] = 999
    write_json(path, data)
    with pytest.raises(ValueError, match="TCN"):
        backend.load_bundle(root, manifest_sha256=rehash(root), runtime=InferenceRuntime(threads=1))


def test_custom_dilations_are_used_and_survive_reload(tmp_path: Path) -> None:
    training, validation = inputs("train"), inputs("validation")
    backend = TCNBackend()
    fitted = backend.fit(
        TrainingInputs(training, np.repeat(np.arange(3), (24, 32, 28)).astype(np.int64)),
        validation,
        parameters=TCNParameters(channels=4, dilations=(1, 3), dropout=0.0),
        policy=policy(),
        analysis=AnalysisConfig(),
        runtime=TrainingRuntime(threads=1),
        evaluator=lambda probabilities: float(probabilities[:, 0].mean()),
    )
    model = cast(TCNPredictor, fitted.predictor).model
    assert len(model.blocks) == 2
    assert [cast(nn.Conv1d, block.conv).dilation for block in model.blocks] == [(1,), (3,)]
    sha = backend.write_bundle(
        fitted,
        tmp_path / "custom",
        provenance=provenance(fitted),
        analysis=fitted.predictor.analysis,
    )
    restored = backend.load_bundle(
        tmp_path / "custom", manifest_sha256=sha, runtime=InferenceRuntime(threads=1)
    )
    np.testing.assert_array_equal(
        restored.predict(validation, runtime=InferenceRuntime(threads=1)),
        fitted.predictor.predict(validation, runtime=InferenceRuntime(threads=1)),
    )


def test_rejects_missing_classes_overlap_ignored_optimizer_and_resource_limits() -> None:
    training, validation = inputs("train"), inputs("validation")
    targets = np.repeat(np.arange(3), (24, 32, 28)).astype(np.int64)

    def run(
        *,
        train: TrainingInputs | None = None,
        valid: ObservationInputs = validation,
        settings: TrainingPolicy | None = None,
        runtime: TrainingRuntime | None = None,
    ) -> None:
        TCNBackend().fit(
            train or TrainingInputs(training, targets),
            valid,
            parameters=TCNParameters(channels=4),
            policy=settings or policy(),
            analysis=AnalysisConfig(),
            runtime=runtime or TrainingRuntime(threads=1),
            evaluator=lambda probabilities: 0.5,
        )

    with pytest.raises(ValueError, match="three"):
        run(train=TrainingInputs(training, np.zeros(len(targets), dtype=np.int64)))
    with pytest.raises(ValueError, match="disjoint"):
        run(valid=training)
    with pytest.raises(ValueError, match="AdamW"):
        run(settings=TrainingPolicy(optimizer="lbfgs", seeds=(11,)))
    with pytest.raises(ValueError, match="32 search"):
        run(settings=TrainingPolicy(optimizer="adamw", seeds=tuple(range(33))))
    with pytest.raises(ResourceBudgetExceeded, match="memory budget"):
        run(runtime=TrainingRuntime(memory_budget_bytes=1))


@pytest.mark.parametrize(
    ("training_lengths", "validation_lengths"),
    [((8, 8, 8), (512,) * 4), ((128, 128, 128), (128,) * 16)],
)
def test_validation_length_and_batch_are_checked_before_feature_allocation(
    monkeypatch: pytest.MonkeyPatch,
    training_lengths: tuple[int, ...],
    validation_lengths: tuple[int, ...],
) -> None:
    import regimex.backends.tcn as tcn

    training = inputs("train", training_lengths)
    validation = inputs("validation", validation_lengths)
    targets = np.repeat(np.arange(3), training_lengths).astype(np.int64)

    def forbidden_features(observations: ObservationInputs) -> PredictionInputs:
        pytest.fail("over-budget validation reached feature allocation")

    monkeypatch.setattr(tcn, "_feature_inputs", forbidden_features)
    with pytest.raises(ResourceBudgetExceeded, match="memory budget"):
        TCNBackend().fit(
            TrainingInputs(training, targets),
            validation,
            parameters=TCNParameters(channels=48),
            policy=TrainingPolicy(optimizer="adamw", seeds=(11,), batch_size=1),
            analysis=AnalysisConfig(),
            runtime=TrainingRuntime(memory_budget_bytes=32 * 1024**2),
            evaluator=lambda _: 0.5,
        )


def test_time_budget_and_inference_cache_budget_fail_explicitly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend, fitted, validation, sha = bundle(tmp_path)
    with pytest.raises(ResourceBudgetExceeded, match="cache budget"):
        backend.load_bundle(
            tmp_path / "bundle",
            manifest_sha256=sha,
            runtime=InferenceRuntime(model_cache_budget_bytes=1),
        )
    with pytest.raises(ResourceBudgetExceeded, match="cache budget"):
        fitted.predictor.predict(validation, runtime=InferenceRuntime(model_cache_budget_bytes=1))
    clock = iter((0.0, 2.0))
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    with pytest.raises(ResourceBudgetExceeded, match="time budget"):
        backend.fit(
            TrainingInputs(inputs("train"), np.repeat(np.arange(3), (24, 32, 28)).astype(np.int64)),
            validation,
            parameters=TCNParameters(channels=4),
            policy=policy(),
            analysis=AnalysisConfig(),
            runtime=TrainingRuntime(time_budget_seconds=1),
            evaluator=lambda probabilities: 0.5,
        )


def test_rejects_compressed_or_duplicate_weights_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend, _, _, _ = bundle(tmp_path)
    root = tmp_path / "bundle"
    with zipfile.ZipFile(root / "weights.pt") as archive:
        contents = {item.filename: archive.read(item) for item in archive.infolist()}
    with zipfile.ZipFile(root / "weights.pt", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("incompatible archive reached torch.load")

    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="archive"):
        backend.load_bundle(root, manifest_sha256=rehash(root), runtime=InferenceRuntime(threads=1))


def test_restricted_weight_loader_rejects_executable_pickle_without_running_it(
    tmp_path: Path,
) -> None:
    backend, _, _, _ = bundle(tmp_path)
    root = tmp_path / "bundle"
    sentinel = tmp_path / "executed"

    class ExecutablePayload:
        def __reduce__(self) -> tuple[object, tuple[str]]:
            return eval, (f"open({str(sentinel)!r}, 'w').write('unsafe')",)

    with zipfile.ZipFile(root / "weights.pt") as archive:
        contents = {item.filename: archive.read(item) for item in archive.infolist()}
    name = next(name for name in contents if name.endswith("/data.pkl"))
    contents[name] = pickle.dumps(ExecutablePayload(), protocol=2)
    with zipfile.ZipFile(root / "weights.pt", "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    with pytest.raises(ValueError, match="restricted TCN"):
        backend.load_bundle(root, manifest_sha256=rehash(root), runtime=InferenceRuntime(threads=1))
    assert not sentinel.exists()
